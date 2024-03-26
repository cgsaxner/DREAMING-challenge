import SimpleITK as sitk
import os
import numpy as np


IMAGE_MASK_MAP = {
    "6c0da39e-2117-4da4-94dd-ba5e2f8188b7": "dcda170d-a3c0-4fd2-af4a-becc2722b3dd", # scene 0092
    "75904266-551a-4985-85d0-8d414151780b": "44c40c43-f866-443b-a848-647e950aebfe", # scene 0096
    "3497b69b-3f26-46da-baa2-45adce1f8a36": "c4c8c33f-139d-4259-864a-4f2b94651216", # scene 100
    "3e39792c-05fe-4c1e-afa5-6135ea84dd6d": "603924b2-c999-4444-a7b0-f6901404b09b", # scene 101
    "defa1da3-9071-44c0-a590-c5df4cb609af": "c5d44730-6a29-4e21-9ca0-1a024603acac", # scene 102
    "53d1f06f-f0bc-4300-a976-d8f38efcb209": "b4021bd9-9117-4a27-9454-5d577fb4d9cc", # scene 103
    "129b1566-55b3-4951-bfaf-3ca4984baa01": "69e8712d-d2db-46ca-88bb-7d81891b1610", # scene 104
    "c28dcfc3-9247-4008-92a2-936717dae3e3": "240d2cec-c040-4406-933d-0d7fe9f4d4cf", # scene 105
    "b114d8c9-512a-4c22-a94d-0acfe64793b8": "93a142a9-96f1-4b3d-b6cc-78ec43250500", # scene 106
    "979c2c13-ebdb-44f6-8225-68b9e6cd8530": "9083c134-f67d-4d15-a9e4-ca3bde30bbc1", # scene 107
    "6ba52586-8517-4ca1-9fde-fb1701f07bc7": "72507319-66b6-4a90-980c-63a5ca2e2269", # scene 108
    "9cc48058-ddee-4596-aa6c-51836859b989": "28075260-e4c2-407d-96ef-fdf5a154c5a7", # scene 109
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