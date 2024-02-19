"""

This will start the evaluation, reads from /input and outputs to /output

"""
import json
from glob import glob
import SimpleITK as sitk
import multiprocessing
from multiprocessing import Pool
from statistics import mean
from pathlib import Path
from pprint import pprint
import os
import numpy as np
from tqdm import tqdm
import torch
_ = torch.manual_seed(123)
from torchmetrics.functional.image.lpips import _NoTrainLpips, _lpips_update, _lpips_compute
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio

from fid import FrechetInceptionDistanceMod as FrechetInceptionDistance

INPUT_DIRECTORY = Path("input")
OUTPUT_DIRECTORY = Path("output")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")
RESOURCES_DIRECTORY = Path("resources")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(INPUT_DIRECTORY / "predictions.json") as f:
        return json.loads(f.read())


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return INPUT_DIRECTORY / job_pk / "output" / relative_path


def load_image_file_as_array(location):
    """
    Load a multi-image mha as a numpy array.
    The output array will have the shape (num_frames, height, width, channels).
    """
    multi_image = sitk.ReadImage(location)
    return sitk.GetArrayFromImage(multi_image)


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(OUTPUT_DIRECTORY / "metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=4))


def process(job):
    # Processes a single algorithm job, looking at the outputs
    report = "Processing:\n Job "
    report += job["pk"]
    # report += pformat(job)
    report += "\n"

    # Firstly, find the location of the results
    inpainted_surgical_scenes_location = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="inpainted-surgical-scenes",
        )
    
    prediction_files = glob(str(inpainted_surgical_scenes_location / "*.mha"))

    results = {
        "pk": job["pk"],
        "mae": 0,
        "psnr": 0,
        "lpips": 0,
    }

    # load the LPIPS model
    print("Loading LPIPS model...")
    lpips_net = _NoTrainLpips(net="alex", pretrained=True, pnet_rand=True,
                              model_path=os.path.join(RESOURCES_DIRECTORY, "alexnet.pth"))
    lpips_net.to(DEVICE)
    print("LPIPS model loaded.")
    
    # load the FID model
    print("Loading FID model...")
    fid_net = FrechetInceptionDistance(feature=2048, normalize=True, 
                                       feature_extractor_weights_path=os.path.join(RESOURCES_DIRECTORY, 
                                                                                   "inception.pth"))
    fid_net.to(DEVICE)
    print("FID model loaded.")

    mean_abolute_error = MeanAbsoluteError().to(DEVICE)
    peak_signal_noise_ratio = PeakSignalNoiseRatio().to(DEVICE)

    prediction_file = prediction_files[0]

    # load predictions
    print(f"Loading prediction file {prediction_file}...")
    prediction_array = load_image_file_as_array(prediction_file)

    # retrieve the input image name to match it with an image in your ground truth
    synthetic_surgical_scenes_image_name = get_image_name(
            values=job["inputs"],
            slug="synthetic-surgical-scenes",
    ) 

    # load ground truth
    # Include it in your evaluation container by placing it in ground_truth/
    scene = synthetic_surgical_scenes_image_name.split(".")[0].split("_")[-1]
    gt_file = os.path.join(GROUND_TRUTH_DIRECTORY, f"gt_scene_{scene}.mha")
    mask_file = os.path.join(GROUND_TRUTH_DIRECTORY, f"mask_scene_{scene}.mha")  
    gt_array = load_image_file_as_array(gt_file)
    mask_array = load_image_file_as_array(mask_file)

    num_frames = prediction_array.shape[0]

    # take timing measurements
    import time
    start = time.time()

    # itereate over the predictions and ground truth
    valid_frames = 0
    for i in tqdm(range(num_frames), desc=f"Processing scene {scene}"):
        # we skip the image if there is no valid mask
        if np.sum(255 - mask_array[i]) == 0:
            continue
        pred_tensor = torch.from_numpy(prediction_array[i]).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)
        gt_tensor = torch.from_numpy(gt_array[i]).unsqueeze(0).permute(0, 3, 1, 2).to(DEVICE)

        results["mae"] += mean_abolute_error(pred_tensor, gt_tensor).item() / 255.0
        results["psnr"] += peak_signal_noise_ratio(pred_tensor, gt_tensor).item()

        pred_tensor_norm = (pred_tensor / 255.0).half()
        gt_tensor_norm = (gt_tensor / 255.0).half()
        
        lpips_loss, lpips_total = _lpips_update(pred_tensor_norm, gt_tensor_norm,
                                                net=lpips_net, normalize=True)
        results["lpips"] += _lpips_compute(lpips_loss.sum(), lpips_total, reduction="mean").item()

        # Update the FID metric
        fid_net.update(pred_tensor_norm, real=False)
        fid_net.update(gt_tensor_norm, real=True)
        valid_frames += 1

    results["mae"] /= valid_frames
    # normalize the mae to the range [0, 1]
    results["mae"] = results["mae"] / 255.0 * 200.0

    results["psnr"] /= valid_frames
    # normalize the psnr to the range [0, 1]
    results["psnr"] = (1 - results["psnr"] / 50.0)

    results["lpips"] /= valid_frames
    # weight the lpips
    results["lpips"] = results["lpips"] * 10.0

    fid = fid_net.compute().item()
    results["fid"] = fid / 100.0

    end = time.time()

    print(f"\nFinished processing scene {scene}.\n" 
          f"Normalized MAE: {results['mae']}, normalized PSNR score: {results['psnr']}, "
          f"Weighted LPIPS: {results['lpips']}, Weighted FID: {results['fid']}. \n"
          f"Time for evaluation run: {end - start} seconds.")

    return results


def main():
    print_inputs()
 
    metrics = {}
    predictions = read_predictions()

    # We now process each algorithm job for this submission
    # Note that the jobs are not in any order!
    # We work that out from predictions.json

    # Start a number of process workers, using multiprocessing
    # The optimal number of workers ultimately depends on how many
    # resources each process() would call upon
    # Need to pass lpips metric to the process function
    multiprocessing.set_start_method("spawn")
    with Pool(processes=2) as pool:
        metrics["results"] = pool.map(process, predictions)

    # Now generate an overall score(s) for this submission
    print("Aggregating results...")
    metrics["aggregates"] = {
        "mae": mean(result["mae"] for result in metrics["results"]),
        "psnr": mean(result["psnr"] for result in metrics["results"]),
        "lpips": mean(result["lpips"] for result in metrics["results"]),
        "fid": mean(result["fid"] for result in metrics["results"]),
    }

    # aggregate psnr and mae to accuracy
    metrics["aggregates"]["accuracy"] = (metrics["aggregates"]["mae"] + metrics["aggregates"]["psnr"]) / 2

    # aggregate lpips and fid to quality
    metrics["aggregates"]["consistency"] = (metrics["aggregates"]["lpips"] + metrics["aggregates"]["fid"]) / 2

    print("Writing metrics...")
    # Make sure to save the metrics
    write_metrics(metrics=metrics)

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
