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


## Author
Mehjabin Rahman Mowrin  
COMP472- F 
Concordia University
