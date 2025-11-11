# CIFAR-10 Decision Tree Classifier (NumPy & Scikit-Learn)

This project implements and evaluates Decision Tree classifiers on the CIFAR-10 dataset using PCA-reduced feature vectors.  
Two versions of the Decision Tree are included:
1. Custom Decision Tree (NumPy from scratch)
2. Scikit-Learn DecisionTreeClassifier implementation
The goal of this project is to compare the performance and behavior of both implementations across different tree depths and analyze the trade-off between underfitting and overfitting.

## Project Structure

├── decision_tree_numpy.py       # Custom NumPy Decision Tree
├── decision_tree_sklearn.py     # Scikit-Learn Decision Tree
├── eval_utils.py                # Metrics + Confusion Matrix Utilities
├── features/                    # PCA-reduced CIFAR-10 feature vectors (not included)
└── results_dt_numpy/ or results_dt_sklearn/ (generated)

## Dataset & Input Features

This project works with 50-dimensional PCA-compressed feature vectors extracted from CIFAR-10 images.
Each feature file must be in `.csv` format with columns:
label, f1, f2, f3, ..., f50
Example folders:
features/train_features.csv
features/test_features.csv

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

For each depth tested, the scripts automatically generate:
cm_depthX.png | Confusion matrix heatmap 
metrics_depthX.json | Accuracy, Precision, Recall, F1-Macro scores 
summary.json | Index of all results for convenience 

Example directory:

results_dt_numpy/
│
├─ cm_depth5.png
├─ cm_depth10.png
├─ cm_depth20.png
├─ cm_depth50.png
│
├─ metrics_depth5.json
├─ metrics_depth10.json
├─ metrics_depth20.json
├─ metrics_depth50.json
│
└─ summary.json

Same structure for:

results_dt_sklearn/
│
├─ cm_depth5.png
├─ cm_depth10.png
├─ cm_depth20.png
├─ cm_depth50.png
│
├─ metrics_depth5.json
├─ metrics_depth10.json
├─ metrics_depth20.json
├─ metrics_depth50.json
│
└─ summary.json

## Key Takeaways
- The NumPy implementation is correct, as its performance closely matches Scikit-Learn.
- Decision Trees rely heavily on depth control to avoid underfitting/overfitting.
- Models like CNNs outperform Decision Trees on image data because they learn *spatial features directly*, not just compressed vector representations.

## Author
Mehjabin Rahman Mowrin  
COMP472- F 
Concordia University

## License
This project is open for educational use.
