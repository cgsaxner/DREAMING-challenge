"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-development-phase | gzip -c > example-algorithm-preliminary-development-phase.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
from glob import glob
import os
import numpy as np
import torch
from torchvision import transforms

from helper import *

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def run():
    # check if cuda is available
    show_torch_cuda_info()

    # Read a resource
    # This can, for example, be your model weights
    with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
        print(f.read())

    # Read the input
    input_location = INPUT_PATH / "images/synthetic-surgical-scenes"
    mask_location = INPUT_PATH / "images/synthetic-surgical-scenes-masks"

    # each scene & corresponding mask is a multi-page tiff
    input_files = glob(str(input_location / "*.tiff")) + glob(str(input_location / "*.mha"))
    mask_files = glob(str(mask_location / "*.tiff")) + glob(str(mask_location / "*.mha"))

    print(f"Found {len(input_files)} input files")
    print(f"Found {len(mask_files)} mask files")

    # iterate over all test scenes
    for i, file in enumerate(input_files):

        # load the image and corresponding mask
        input_id = get_scene_id(file)
        mask_id = IMAGE_MASK_MAP[input_id]

        print(f"Processing scene {input_id}...")

        input_array = load_image_file_as_array(file)
        mask_array = load_image_file_as_array(os.path.join(mask_location, 
                                                           f"{mask_id}.mha"))

        # Process the inputs: any way you'd like
        # For now, let us set make bogus predictions
    
        # convert masks to binary
        mask_array = mask_array > 0

        # produce a masked images
        input_masked = input_array * mask_array

        # Create bogus inpainted images; just ones for now
        # run your forward pass here
        inpainted_array = np.ones_like(input_masked) * 255

        # Save the output
        write_array_as_image_file(
            location=os.path.join(OUTPUT_PATH, "images", 
                                  "inpainted-synthetic-surgical-scenes"),
            scene_id=input_id,    
            array=inpainted_array,
        )
    
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
