# Image Captioning Project

## Overview
This project aims to develop a neural network architecture for automatically generating captions from images based on the paper [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf). The system leverages Convolutional Neural Networks (CNNs) to process images and Recurrent Neural Networks (RNNs) to generate descriptive text captions. The model utilizes a CNN to encode the image into a fixed-length vector representation and an RNN to decode this vector into a descriptive sentence. The Microsoft Common Objects in Context ([MS COCO](http://cocodataset.org/#home)) dataset is utilized for training the model. This README provides detailed instructions on setting up and running the project, highlighting the integration of advanced computer vision and natural language processing techniques.
![Neural Image Caption, or NIC model](https://github.com/hhosseinian/ImageCaptioning/blob/main/Images/Image_Captioning_Arch.png)

## Table of Contents
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Prerequisites
Before running the project, ensure you have the following prerequisites installed:

    * Python (>=3.6)
    * PyTorch (>=0.4)
    * NumPy
    * Matplotlib
    * Jupyter Notebook (for running the provided notebooks)

You can install most of these dependencies using pip or conda.

**Important**: GPU mode must be enabled for this project. Training a fully functional model on a GPU is expected to take between 5-12 hours. It is recommended to monitor early patterns in loss during the first hour of training as you make adjustments to your model. This approach helps minimize the time spent on extensive training until you are ready to train your final model.


## Project Structure
The project directory is organized as follows:
- **0_Dataset.ipynb**: This notebook is for exploring the Microsoft COCO dataset that will be used for training the image-capturing architecture. You can explore how to Initialize the COCO API and print a sample image, along with its five corresponding captions. You can read more about the dataset on ([MS COCO](http://cocodataset.org/#home)) website or in the [research paper](https://arxiv.org/pdf/1405.0312).
- **1_Preliminaries**: In this notebook, we load and pre-process data from the COCO dataset. We further design a CNN-RNN architecture/model for automatically generating image captions. 
- **2_Training**: In this notebook, we train the CNN_RNN model developed in previous step. 
- **3_Inference**: In this notebook, we use the trained model to generate captions for images in the test dataset.

## Usage
### Training

To train the network, follow these steps:

    - 0_Dataset.ipynb:
        Load and explore the MS COCO dataset.
        Initialize the COCO API and print a sample image with its corresponding captions.
    - 1_Preliminaries.ipynb:
        Pre-process the data from the COCO dataset.
        Design and develop the CNN-RNN model.
    - 2_Training.ipynb:
        Train the CNN-RNN model using the pre-processed data.
        Adjust model parameters as needed and monitor loss patterns.

### Suggestions for modifying the model:

    Experiment with different architectures for the CNN and RNN components.
    Try varying the size of the hidden layers or the number of layers in the RNN.
    Adjust hyperparameters such as learning rate, batch size, and number of epochs to observe their impact on model performance.

### Inference

To see the model in action:

    - 3_Inference.ipynb:
        Use the trained model to generate captions for images in the test dataset.
        Evaluate model performance by comparing generated captions to the actual captions.

Steps for inference:

    - Load the trained model.
    - Pass a test image through the model to generate a caption.
    - Compare the generated caption with the actual caption to evaluate performance.



## Results (Under revision)
  -provide results for your specific example.
  -provide results when you change different hyperparameters


## Troubleshooting `ModuleNotFoundError` for `pycocotools` in running **0_Dataset.ipynb**

If you encounter a `ModuleNotFoundError` for `pycocotools` in your Python environment, follow these steps to resolve the issue:

### 1. Install `pycocotools`

First, ensure that `pycocotools` is installed in your Python environment. You can do this using pip:

```bash
pip install pycocotools

After installation, verify that pycocotools is installed correctly and can be imported without errors:

import pycocotools

If no errors occur, the package is installed correctly.

Verify Module Structure

Ensure that the structure of the pycocotools module is correct. It should have a coco.py file within a directory named pycocotools. The import statement from pycocotools.coco import COCO expects this structure to be present.
5. Restart Kernel or Script Execution

If you installed pycocotools after starting your Python session or script, restart the Python kernel or script execution to ensure that the changes take effect.

By following these steps, you should be able to resolve the ModuleNotFoundError for pycocotools and successfully use the package in your Python projects.


Feel free to adjust the paths and instructions as per your specific setup and requirements. This Markdown can be directly added to your GitHub README file to help others troubleshoot similar issues with `pycocotools`.


## Acknowledgments
- This project is part of the [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891) program from Udacity.
- The project's architecture and data loader are based on research papers and materials provided by Udacity.
