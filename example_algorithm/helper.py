import SimpleITK as sitk
import os
import numpy as np


IMAGE_MASK_MAP = {
    "6c0da39e-2117-4da4-94dd-ba5e2f8188b7": "dcda170d-a3c0-4fd2-af4a-becc2722b3dd", # scene 0092
    "75904266-551a-4985-85d0-8d414151780b": "44c40c43-f866-443b-a848-647e950aebfe", # scene 0096
}

def show_torch_cuda_info():
    """
    Show some information about the available CUDA devices using torch.
    """
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


def get_scene_id(file_path):
    return file_path.split("/")[-1].split(".")[0].split("_")[-1]


def load_image_file_as_array(location):
    """
    Load a multi-image mha as a numpy array.
    The output array will have the shape (num_frames, height, width, channels).
    """
    multi_image = sitk.ReadImage(location)
    array = sitk.GetArrayFromImage(multi_image)
    if array.ndim < 4:
        array = np.expand_dims(array, axis=3)
    return array

def write_array_as_image_file(location, scene_id, array):
    """
    Write a numpy array to a multi-image mha file.
    The input array should have the shape (num_frames, height, width, channels).
    Individual images should be of UINT type in range [0, 255].
    """
    create_dir(location)

    image = sitk.GetImageFromArray(array)
    sitk.WriteImage(image, os.path.join(location, f"{scene_id}.mha"), 
                    useCompression=True)
    
    print(f"Saved {location}")


def display_sample(inputs, masks, predictions):
    """
    Helper function to display input, mask and prediction array using matplotlib 
    """
    import matplotlib.pyplot as plt
    for i in range(inputs.shape[0]):
        plt.figure()
        plt.subplot(311)
        plt.imshow(inputs[i])
        plt.subplot(312)
        plt.imshow(masks[i])
        plt.subplot(313)
        plt.imshow(predictions[i])
        plt.show()


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)