# CIFAR-10 Image Classification using VGG11 CNN

## Overview

This project implements a **VGG11 Convolutional Neural Network (CNN)** for image classification on the **CIFAR-10 dataset**.
The model is designed to classify small 32×32 color images into **10 object categories** such as airplanes, cars, birds, cats, and ships.
It uses **PyTorch** for model definition, training, and evaluation.


## Model Architecture

The architecture follows the **VGG11** design:

* Sequential convolutional layers with small 3×3 kernels
* ReLU activations
* Max-pooling layers for down-sampling
* Fully connected layers for classification
* 

## Training Details

* **Dataset:** CIFAR-10
* **Framework:** PyTorch
* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss
* **Learning Rate:** 0.001
* **Batch Size:** 64
* **Epochs:** 30
* **Hardware:** Trained on GPU (recommended)
  

## Results

| Metric        | Value                            |
| :------------ | :------------------------------- |
| Accuracy      | ~84–86%                          |
| Loss          | Decreases steadily across epochs |
| Dataset Split | 80% Training / 20% Testing       |

   

## Project Structure

SOEN472/
│
├── models/                # Saved model files (.pth)
│
├── vgg11_cnn.py           # Main model architecture and training code
├── eval_utils.py          # Utility functions for evaluation and metrics
├── README.md              # Project documentation
├── .gitattributes         # Git LFS tracking
└── .gitignore.txt         # Ignored files and directories



## How to Run

1. Clone the repository:
   git clone https://github.com/<your-username>/SOEN472.git
   cd SOEN472
  
2. Run training:
   python vgg11_cnn.py

3. Evaluate model:
   python eval_utils.py

4. The trained model will be saved in the `models/` folder as:
   vgg11_main.pth


## Key Features
* Implements **VGG11 CNN** using PyTorch
* Handles training, validation, and testing automatically
* Provides evaluation utilities for accuracy and a confusion matrix
* Compatible with GPU acceleration


## Author
Mehjabin Rahman Mowrin
Comp472- F
Concordia University

