
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import itertools
import json
import os

CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    per_cls = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "per_class": {
            "precision": per_cls[0].tolist(),
            "recall": per_cls[1].tolist(),
            "f1": per_cls[2].tolist()
        }
    }

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CIFAR10_CLASSES))
    plt.xticks(tick_marks, CIFAR10_CLASSES, rotation=45, ha="right")
    plt.yticks(tick_marks, CIFAR10_CLASSES)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return out_path

def save_metrics_json(metrics, out_json):
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    return out_json
