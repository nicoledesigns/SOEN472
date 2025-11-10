#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ---------- Class-name helpers (CIFAR-10) ----------
DEFAULT_CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_class_names(map_json_path=None, n_classes=10):
    """
    Return list of class names ordered by numeric id 0..n-1.
    If map_json_path is provided, it should be a mapping name->id (e.g. {"airplane":0,...}).
    We invert it to id->name; otherwise fall back to CIFAR-10 default names.
    """
    if map_json_path is None:
        return DEFAULT_CIFAR10_CLASSES[:n_classes]
    try:
        with open(map_json_path, "r") as f:
            m = json.load(f)            # name -> id
        inv = {v: k for k, v in m.items()}  # id -> name
        return [inv[i] for i in range(n_classes)]
    except Exception:
        return DEFAULT_CIFAR10_CLASSES[:n_classes]

# ---------- Metrics (NumPy) ----------
def confusion_matrix_np(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm

def precision_recall_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1        = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)

    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()

    accuracy = tp.sum() / cm.sum()
    micro_p = micro_r = micro_f1 = accuracy  # single-label multiclass

    return {
        "per_class_precision": precision.tolist(),
        "per_class_recall":    recall.tolist(),
        "per_class_f1":        f1.tolist(),
        "macro_precision": float(macro_p),
        "macro_recall":    float(macro_r),
        "macro_f1":        float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall":    float(micro_r),
        "micro_f1":        float(micro_f1),
        "accuracy":        float(accuracy),
    }

def print_summary(y_true, y_pred, header="Model"):
    cm = confusion_matrix_np(y_true, y_pred)
    m = precision_recall_f1_from_cm(cm)
    print(f"\n===== {header} =====")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nAccuracy: {:.4f}".format(m["accuracy"]))
    print("Macro Precision: {:.4f} | Macro Recall: {:.4f} | Macro F1: {:.4f}".format(
        m["macro_precision"], m["macro_recall"], m["macro_f1"]))
    print("Micro Precision/Recall/F1 (== accuracy for single-label multiclass): {:.4f}".format(
        m["micro_f1"]))
    return m, cm

# ---------- Save reports ----------
def _timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def save_report(model_name, y_true, y_pred, out_dir="reports", class_names=None):
    os.makedirs(out_dir, exist_ok=True)
    base = f"{model_name.replace(' ', '_').lower()}_{_timestamp()}"

    cm = confusion_matrix_np(y_true, y_pred)
    metrics = precision_recall_f1_from_cm(cm)
    metrics["model"] = model_name
    metrics["n_classes"] = int(cm.shape[0])

    # metrics JSON
    metrics_path = os.path.join(out_dir, f"{base}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix CSV (labeled if names provided)
    if class_names and len(class_names) == cm.shape[0]:
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    else:
        cm_df = pd.DataFrame(cm)

    cm_path = os.path.join(out_dir, f"{base}_confusion_matrix.csv")
    cm_df.to_csv(cm_path, index=True, header=True)

    print(f"[saved] {metrics_path}")
    print(f"[saved] {cm_path}")

def save_text_report(model_name, y_true, y_pred, class_names, out_dir="reports"):
    """Pretty, labeled .txt report with metrics + confusion matrix."""
    os.makedirs(out_dir, exist_ok=True)
    base = f"{model_name.replace(' ', '_').lower()}_{_timestamp()}"
    path = os.path.join(out_dir, f"{base}_report.txt")

    cm = confusion_matrix_np(y_true, y_pred, n_classes=len(class_names))
    m  = precision_recall_f1_from_cm(cm)

    with open(path, "w") as f:
        f.write(f"{'='*10} {model_name} {'='*10}\n\n")
        f.write("Confusion Matrix (rows=true, cols=pred):\n")

        # Header row (truncate names to keep table narrow)
        header = " " * 12 + " ".join([f"{c[:8]:>8}" for c in class_names]) + "\n"
        f.write(header)

        # Rows
        for i, cname in enumerate(class_names):
            row = " ".join(f"{v:8d}" for v in cm[i])
            f.write(f"{cname[:10]:>10} {row}\n")

        f.write("\n")
        f.write(f"Accuracy: {m['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {m['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:    {m['macro_recall']:.4f}\n")
        f.write(f"Macro F1:        {m['macro_f1']:.4f}\n")
        f.write(f"Micro F1 (Accuracy): {m['micro_f1']:.4f}\n")
        f.write("\nPer-Class F1 Scores:\n")
        for cname, f1 in zip(class_names, m["per_class_f1"]):
            f.write(f"  {cname:10s}: {f1:.4f}\n")

    print(f"[saved formatted text report] {path}")

# ---------- NumPy-only Gaussian Naive Bayes ----------
class GaussianNB_Numpy:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.theta_ = None
        self.sigma_ = None
        self.class_log_prior_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        C, D = len(self.classes_), X.shape[1]

        self.theta_ = np.zeros((C, D), dtype=np.float64)
        self.sigma_ = np.zeros((C, D), dtype=np.float64)
        class_count = np.zeros(C, dtype=np.float64)

        eps = self.var_smoothing * np.var(X, axis=0).max()

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            class_count[i] = Xc.shape[0]
            self.theta_[i] = Xc.mean(axis=0)
            var = Xc.var(axis=0) + eps
            var[var == 0.0] = eps
            self.sigma_[i] = var

        priors = class_count / class_count.sum()
        priors = np.where(priors == 0.0, 1e-12, priors)
        self.class_log_prior_ = np.log(priors)
        return self

    def _joint_log_likelihood(self, X):
        X = np.asarray(X, dtype=np.float64)            # (N, D)
        log_term = np.log(2.0 * np.pi * self.sigma_)   # (C, D)
        diff2_over_var = (X[:, None, :] - self.theta_[None, :, :])**2 / self.sigma_[None, :, :]
        # sum over features -> (N, C)
        sum_ll = -0.5 * (np.sum(log_term[None, :, :], axis=2) + np.sum(diff2_over_var, axis=2))
        return sum_ll + self.class_log_prior_[None, :] # (N, C)

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

# ---------- IO ----------
def load_csv(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(np.float64)
    y = df["label"].values.astype(int)
    return X, y

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Gaussian Naive Bayes on CIFAR-10 feature CSVs")
    ap.add_argument("--train", required=True, help="Path to train_features.csv")
    ap.add_argument("--test",  required=True, help="Path to test_features.csv")
    ap.add_argument("--smooth", type=float, default=1e-9, help="Variance smoothing (NumPy & scikit)")
    ap.add_argument("--sklearn", action="store_true", help="Also run scikit-learn GaussianNB")
    ap.add_argument("--out", default="reports", help="Folder to save metrics/CM/text reports")
    ap.add_argument("--classmap", default=None,
                    help="Optional path to class_to_idx.json for labeling the confusion matrix. "
                         "If omitted, uses standard CIFAR-10 names.")
    args = ap.parse_args()

    # Load data
    X_train, y_train = load_csv(args.train)
    X_test,  y_test  = load_csv(args.test)

    # Class names (custom map or CIFAR-10 defaults)
    class_names = load_class_names(args.classmap, n_classes=10)

    # Part 1: NumPy-only GNB
    gnb_np = GaussianNB_Numpy(var_smoothing=args.smooth).fit(X_train, y_train)
    yhat_np = gnb_np.predict(X_test)
    print_summary(y_test, yhat_np, header="NumPy GaussianNB")
    save_report("NumPy GaussianNB", y_test, yhat_np, out_dir=args.out, class_names=class_names)
    save_text_report("NumPy GaussianNB", y_test, yhat_np, class_names, out_dir=args.out)

    # Part 2: scikit-learn GNB (optional)
    if args.sklearn:
        try:
            from sklearn.naive_bayes import GaussianNB
            gnb_sk = GaussianNB(var_smoothing=args.smooth)
            gnb_sk.fit(X_train, y_train)
            yhat_sk = gnb_sk.predict(X_test)
            print_summary(y_test, yhat_sk, header="scikit-learn GaussianNB")
            save_report("scikit-learn GaussianNB", y_test, yhat_sk,
                        out_dir=args.out, class_names=class_names)
            save_text_report("scikit-learn GaussianNB", y_test, yhat_sk,
                             class_names, out_dir=args.out)
        except Exception as e:
            print("\n[warn] scikit-learn not available or failed:", e)
            print("Install with: pip install scikit-learn")

if __name__ == "__main__":
    main()
    