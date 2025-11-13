
"""
Scikit-learn Decision Tree for COMP472 - CIFAR-10 features (50-d PCA).
- Gini, max_depth sweep
- Evaluation with confusion matrix & metrics

# Depth variations tested during experimentation:
# for depth in [5, 10, 20, 50]:
#     clf = DecisionTreeClassifier(max_depth=depth)
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_test)
#     print(f"Depth {depth}, Accuracy: {accuracy_score(y_test, preds)}")

Usage:
  python decision_tree_sklearn.py --train features/train_features.csv --test features/test_features.csv --outdir results_dt_sklearn --depths 5 10 20 50
"""
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from eval_utils import compute_metrics, plot_confusion, save_metrics_json

def load_features(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="results_dt_sklearn")
    ap.add_argument("--depths", type=int, nargs="+", default=[50])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    Xtr, ytr = load_features(args.train)
    Xte, yte = load_features(args.test)

    results = {}
    for d in args.depths:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        metrics = compute_metrics(yte, yhat)
        cm_path = os.path.join(args.outdir, f"cm_depth{d}.png")
        plot_confusion(yte, yhat, cm_path)
        out_json = os.path.join(args.outdir, f"metrics_depth{d}.json")
        save_metrics_json(metrics, out_json)
        results[f"depth_{d}"] = {"metrics_json": out_json, "confusion_png": cm_path}

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Done. Results in", args.outdir)

if __name__ == "__main__":
    main()
