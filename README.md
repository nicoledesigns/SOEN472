# SOEN472
#Members:
Nayla Nocera 40283927
Mehjabin Rahman Mowrin 
# Gaussian Naive Bayes — CIFAR-10 Feature Classification

This project implements **Gaussian Naive Bayes (GNB)** for classifying CIFAR-10 images using **pre-extracted feature vectors**.  
The script trains the model, evaluates it, generates reports, and saves the trained models.
---

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
