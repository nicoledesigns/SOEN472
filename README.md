# CIFAR-10 Image Classification using VGG11 CNN
This project implements and evaluates a VGG11 Convolutional Neural Network for CIFAR-10 image classification.  
The model is trained directly on 32×32 RGB images and learns hierarchical spatial features through convolution, pooling, batch normalization, and dropout.

## Model Architecture

The architecture follows the VGG11 specification, adapted for CIFAR-10:

- Convolution → BatchNorm → ReLU layers
- Five MaxPooling layers to reduce spatial size
- Feature tensor (512) flattened to a fully connected classifier:
  - 4096 → 4096 → 10 output classes
- Dropout (0.5) to reduce overfitting

> Implemented in: `VGG11` class inside vgg11_cnn.py:contentReference[oaicite:2]{index=2}

## Dataset

This model uses the CIFAR-10 dataset loaded automatically via `torchvision.datasets.CIFAR10`.  
It contains 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Data Augmentation applied (training only):
- Random Horizontal Flip  
- Normalization to CIFAR-10 mean and std

## Training & Evaluation

### Train the Model
python vgg11_cnn.py --mode train --epochs 30 --batch_size 128 --lr 0.01 --outdir results_vgg11

### Evaluate a Saved Model
python vgg11_cnn.py --mode eval --checkpoint models/vgg11_main.pth --outdir results_vgg11_eval

**Kernel Size Variation**
receptive fields by changing:

--kernel_size 2
--kernel_size 3       (default/recommended)
--kernel_size 5
--kernel_size 7

## Output Files

The script automatically generates performance artifacts:

| File                    | Purpose                               |
| ----------------------- | ------------------------------------- |
| `cm_vgg11_k*.png`       | Confusion matrix heatmap              |
| `metrics_vgg11_k*.json` | Accuracy, Precision, Recall, F1-Score |
| `vgg11_main.pth`        | Saved trained model checkpoint        |

Example output directory:

results_vgg11/
│
├─ cm_vgg11_k3.png
├─ metrics_vgg11_k3.json
│
└─ models/vgg11_main.pth

> Output is produced using utilities in **eval_utils.py** (metrics + confusion matrices). 


## Short Summary 

Trained a VGG11 CNN on CIFAR-10 and compared kernel-size variations. The best performance was achieved with a kernel size of 3, demonstrating strong generalization through hierarchical feature learning and effective spatial pattern extraction.

## Author
Mehjabin Rahman Mowrin
Comp472- F
Concordia University

