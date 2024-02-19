# DREAMING challenge evaluation script

This is the evaluation script used for the [DREAMING challenge](https://dreaming.grand-challenge.org/). 

## Requirements

* This repository requires [PyTorch](https://pytorch.org/) with torchvision.
* After installing PyTorch for your system, install the additional requirements using
  ```
  pip install -r requirements.txt
  ```

## Inputs and outputs
* The script expects the output of a grand-challenge.org submission run as input.
  * Individual run output folders should be placed under the [input](/input/) folder.
  * An example [predictions.json](/input/predictions.json) is included in the repository.
  * If you want to test the algorithm using other inputs, please adapt the code accordingly.
* It will produce a metrics.json file as output.