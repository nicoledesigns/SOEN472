# CIFAR-10 Image Classification using VGG11 CNN
This project implements and evaluates a VGG11 Convolutional Neural Network for CIFAR-10 image classification.  
The model is trained directly on 32×32 RGB images and learns hierarchical spatial features through convolution, pooling, batch normalization, and dropout.

## Model Architecture
The network follows the VGG11 design adapted for small CIFAR-10 inputs.  
It stacks sequential convolution → BatchNorm → ReLU blocks, with MaxPooling to reduce spatial size, and a fully connected classification head.

Input (32×32×3)
↓
Conv → BN → ReLU → MaxPool
↓
Conv → BN → ReLU → MaxPool
↓
Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
↓
Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
↓
Conv → BN → ReLU → Conv → BN → ReLU → MaxPool
↓
Flatten → FC(4096) → ReLU → Dropout(0.5)
↓
FC(4096) → ReLU → Dropout(0.5)
↓
Output Layer (10 classes)

The convolution kernel size is configurable (`2`, `3`, `5`, or `7`) to study receptive field effects.

## Dataset

This model trains on CIFAR-10, consisting of 10 object categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
The dataset is loaded automatically using `torchvision.datasets.CIFAR10`.

## Training and Evaluation

### Train the CNN
python vgg11_cnn.py --mode train --epochs 30 --batch_size 128 --lr 0.01 --outdir results_vgg11

### Evaluate a Saved Model
python vgg11_cnn.py --mode eval --checkpoint models/vgg11_main.pth --outdir results_vgg11_eval

## Output Files

After training/evaluating, the script automatically generates:

| Output File             | Description                           |
| ----------------------- | ------------------------------------- |
| `cm_vgg11_k*.png`       | Confusion matrix heatmap              |
| `metrics_vgg11_k*.json` | Accuracy, Precision, Recall, F1-Score |
| `vgg11_main.pth`        | Saved trained model weights           |

Example directory:

results_vgg11/
│
├─ cm_vgg11_k3.png
├─ metrics_vgg11_k3.json
│
└─ models/vgg11_main.pth

## Key Observations

* Kernel size = 3 gives the best balance of feature detail and stability.
* Larger kernels capture broader shape context but may smooth away fine texture details.
* CNN significantly outperforms Decision Trees and Naive Bayes because it learns spatial structure directly from pixels.

## Short Summary 

Trained a VGG11 CNN on CIFAR-10 and compared kernel-size variations. The best performance was achieved with a kernel size of 3, demonstrating strong generalization through hierarchical feature learning and effective spatial pattern extraction.

## Author
Mehjabin Rahman Mowrin
Comp472- F
Concordia University

