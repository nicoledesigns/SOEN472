# SOEN472
#Members:
Nayla Nocera 40283927
Mehjabin Rahman Mowrin 40223512 

# Gaussian Naive Bayes — CIFAR-10 Feature Classification

This project implements **Gaussian Naive Bayes (GNB)** for classifying CIFAR-10 images using **pre-extracted feature vectors**.  
The script trains the model, evaluates it, generates reports, and saves the trained models.

## Dataset

You work directly with:

- `train_features.csv`
- `test_features.csv`

Each contains:

- 50 numerical features (`f1` … `f50`)
- A `label` column
- Labels correspond to CIFAR-10 classes: airplane, automobile, bird, cat, deer,dog, frog, horse, ship, truck

## Setup Instructions

### 1. Create a virtual environment
```bash
python3 -m venv venv

---
### 2. Activate the environemnt 
source venv/bin/activate

---
###3. Install dependencies
pip install -r requirements.txt

###4. Run NumPy Gaussian Naive Bayes
python3 scripts/naive_bayes.py \
  --train features/train_features.csv \
  --test  features/test_features.csv

###5. NumPy + scikit-learn GaussianNB
python3 scripts/naive_bayes.py \
  --train features/train_features.csv \
  --test  features/test_features.csv \
  --sklearn
```

Trained models at: 
reports/numpy_gaussiannb_model.npz
reports/sklearn_gaussiannb_model.joblib

Metrics (JSON):
*_metrics.json

Confusion Matrix
*_confusion_matrix.csv

Formatted Evaluation Report (TXT)
*_report.txt


# CIFAR-10 Decision Tree Classifier (NumPy & Scikit-Learn)

This project implements and evaluates Decision Tree classifiers on the CIFAR-10 dataset using PCA-reduced feature vectors.  
Two versions of the Decision Tree are included:
1. Custom Decision Tree (NumPy from scratch)
2. Scikit-Learn DecisionTreeClassifier implementation
The goal of this project is to compare the performance and behavior of both implementations across different tree depths and analyze the trade-off between underfitting and overfitting.

## Project Structure

### File and Folder Overview

1. **`decision_tree_numpy.py`** → Custom Decision Tree implementation built entirely with NumPy.
2. **`decision_tree_sklearn.py`** → Decision Tree implementation using the Scikit-Learn library.
3. **`eval_utils.py`** → Contains helper functions for generating evaluation metrics and confusion matrix plots.
4. **`features/`** → Stores PCA-reduced CIFAR-10 feature vectors (not included in the repository).
5. **`models/`** → Contains saved model files such as `.pkl` (scikit-learn models) or `.pth` (PyTorch models).
6. **`results_dt_numpy/`** or **`results_dt_sklearn/`** → Automatically generated output folders containing:
   * `cm_depthX.png` → Confusion matrix heatmap for each tested depth
   * `metrics_depthX.json` → Accuracy, Precision, Recall, and F1-Macro scores
   * `summary.json` → Summary index of all metrics for convenience

## Dataset

This project works with 50-dimensional PCA-compressed feature vectors extracted from CIFAR-10 images.
Each feature file must be in `.csv` format with columns:
label, f1, f2, f3, ..., f50
Example folders:
1. features/train_features.csv
2. features/test_features.csv

## Running the Models

1. NumPy Decision Tree
--python decision_tree_numpy.py
--train features/train_features.csv 
--test features/test_features.csv 
--outdir results_dt_numpy 
--depths 5 10 20 50

2. Scikit-Learn Decision Tree
--python decision_tree_sklearn.py
--train features/train_features.csv 
--test features/test_features.csv 
--outdir results_dt_sklearn 
--depths 5 10 20 50

## Generated Output

### Output Files Generated for Each Depth Tested

For every tested tree depth, the script automatically creates:

1. **`cm_depthX.png`** → Confusion matrix heatmap visualization
2. **`metrics_depthX.json`** → Stores the evaluation metrics:

   * Accuracy
   * Precision
   * Recall
   * F1-Macro score
3. **`summary.json`** → Collects and indexes all metric results for convenience

### Example Folder Structure

#### **NumPy Implementation (`results_dt_numpy/`)**

* Confusion matrix images: `cm_depth5.png`, `cm_depth10.png`, `cm_depth20.png`, `cm_depth50.png`
* Metrics JSON files: `metrics_depth5.json`, `metrics_depth10.json`, `metrics_depth20.json`, `metrics_depth50.json`
* Summary file: `summary.json`

#### **scikit-learn Implementation (`results_dt_sklearn/`)**

* Confusion matrix images: `cm_depth5.png`, `cm_depth10.png`, `cm_depth20.png`, `cm_depth50.png`
* Metrics JSON files: `metrics_depth5.json`, `metrics_depth10.json`, `metrics_depth20.json`, `metrics_depth50.json`
* Summary file: `summary.json`


## Key Takeaways
- The NumPy implementation is correct, as its performance closely matches Scikit-Learn.
- Decision Trees rely heavily on depth control to avoid underfitting/overfitting.
- Models like CNNs outperform Decision Trees on image data because they learn *spatial features directly*, not just compressed vector representations.


MLP model:

1.
* Created and activated a clean isolated Python environment.
* Installed all necessary data science and ML dependencies.
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision scikit-learn numpy pandas matplotlib tqdm seaborn

2. Recreate my exact setup with:
pip install -r requirements.txt

3. Run python scripts/make_csv_features.py
4. Do : source venv/bin/activate
5. Run: python mlp_model.py

MLP files:
mlp_mode.py : contains the MLP implementation with different variants (depths and layers) and creates a confusion matrix for each model based on the base model. 
mlp_depth_summary (layer variants): contains the model name, Accuracy, Precision, Recall, F1 for each 1 layer and 2 layer mlp models.
mlp_variants_summary (size variants): contains the model name, Accuracy, Precision, Recall, F1 for each narrow, wide and extra wide mlp models.
requirements.txt: recreate the python environmnent to test the mlp model.

mlp_1layer.pth, mlp_3layer.pth, mlp_narrow.pth, mlp_wide.pth, mlp_extra_wide.pth, mlp_base.pth: are the saved models
  
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
* **models/** – Folder that stores saved model files.
* **vgg11_cnn.py** – Main CNN model and training code.
* **eval_utils.py** – Functions for evaluation and metrics.
* **README.md** – Project description.
* **.gitattributes** – Git LFS settings.
* **.gitignore.txt** – Files/folders Git should ignore.

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
