# DREAMING challenge example algorithm

This is an example algorithm for the [DREAMING challenge](https://dreaming.grand-challenge.org/). You can use this repository to create your submission for the challenge.

## Requirements

* This repository requires [Docker](https://www.docker.com/).
* If you are using Windows, we recommend Docker with Windows Subsystem for Linux (WSL) 2 backend.
  * Installation instructions for Docker on Windows are found in the [Docker documentation](https://docs.docker.com/desktop/install/windows-install/).
  * For setting up WSL 2 refer to the [Microsoft documentation](https://learn.microsoft.com/en-us/windows/wsl/install).
  * Grand-challenge.org also provides [Documentation](https://grand-challenge.org/documentation/setting-up-wsl-with-gpu-support-for-windows-11/) for setting up Docker with GPU support on Windows.

## How to use

* Prepare your inference algorithm. Start from the `inference.py` file, which is the default executable for the Docker container.
  * Note: Some parts of the grand-challenge.org documentation mention using evalutils, e.g. [here](https://grand-challenge.org/documentation/prepare-your-code-for-containerization/). We do not use evalutils for this challenge and you can ignore it.
* Modify the `Dockerfile`. You could
  * Change the [Docker base container](https://hub.docker.com/) to fit your environment. This example uses a PyTorch container.
  * If you add any files to your algorithm, make sure that the `Dockerfile` copies them by adding
    ```
    COPY --chown=user:user <path/to/your_file> /opt/app/<path/to/your_file> 
    ```
* Modify the `requirements.txt` file. Add all requirements of your code that are not contained in your Docker base container.
* You can test your container locally by running: 
    ```
    ./test_run.sh
    ```
* To prepare your algorithm for submission, build the Docker container by running:
    ```
    docker save example-algorithm-preliminary-development-phase | gzip -c > example-algorithm-preliminary-development-phase.tar.gz
    ```
* Submit your Algorithm container to the challenge on the [DREAMING submission site](https://dreaming.grand-challenge.org/evaluation/preliminary-development-phase/submissions/create/).
  * Create a grand-challenge.org Algorithm from your Docker container: [Create an Algorithm page](https://grand-challenge.org/documentation/create-an-algorithm-page/), then [upload your container image](https://grand-challenge.org/documentation/exporting-the-container/).
  * [Submit the container to the Challenge](https://grand-challenge.org/documentation/making-a-challenge-submission/#submitting-your-algorithm-container).
<!-- * Alternatively, you can create your own private Github repository and link it to an Algorithm submssion on grand-challenge.org
  * [Creating a new Github repository from a challenge example algorithm](https://grand-challenge.org/documentation/clone-a-repository-from-a-challenge-baseline-algorithm/)
  * [Linking a Github repository to a grand-challenge.org Algorithm](https://grand-challenge.org/documentation/linking-a-github-repository-to-your-algorithm/) -->

## Inputs and outputs

### Inputs
* Each testing scene and corresponding mask is stored on grand-challenge.org as a multi-image .mha file. Examples are provided in the [test image](test/input/images/synthetic-surgical-scenes/) and [test mask](test/input/images/synthetic-surgical-scenes-masks/) folders.
* The files have the shape `[num_frames, height, width, channels]`, where `channels` is 3 for input scenes, and 1 for input masks.
* The function `load_image_file_as_array` in `helper.py` shows how to read input files and returns them UINT8 numpy arrays of images in the range [0, 255].
* Feel free to extend this function to fit the inputs expected by your algorithm.

### Outputs
* The output of the algorithm should, again, be a multi-image .mha file per testing scene.
* The function `write_array_as_image_file` in `helper.py` shows how to write a UINT8 numpy array of shape `[num_frames, height, width, channels]` containing images in the range [0, 255] to a valid output file.
* If you use other image formats in your algorithm, e.g. float in range [0, 1] or [-1, 1], please convert your predictions to the correct format before passing them to `write_array_as_image_file`. This ensures that metrics will be computed correctly.