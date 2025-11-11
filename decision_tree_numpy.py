
"""
Decision Tree (from scratch, NumPy) for COMP472 - CIFAR-10 features (50-d PCA).
- Gini impurity
- Max depth (default=50)
- Min samples split/pruning knobs included
- Experiments over depths and proper evaluation/CM figure

# Depth variations tested during experimentation:
# depths = [5, 10, 20, 50]
# for d in depths:
#     tree = build_tree(X_train, y_train, max_depth=d)
#     y_pred = predict(tree, X_test)
#     print(f"Depth {d} Results: {compute_metrics(y_test, y_pred)}")


Usage:
  python decision_tree_numpy.py --train features/train_features.csv --test features/test_features.csv --outdir results_dt_numpy --depths 5 10 20 50
"""
import argparse, os, json
import numpy as np
import pandas as pd
from eval_utils import compute_metrics, plot_confusion, save_metrics_json, CIFAR10_CLASSES

class TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "is_leaf", "prediction")
    def __init__(self, is_leaf=False, prediction=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p**2)

def best_split(X, y):
    # Returns (feature_idx, threshold, left_indices, right_indices)
    n_samples, n_features = X.shape
    base_gini = gini_impurity(y)
    best_feat, best_thr, best_gain = None, None, 0.0
    best_left_idx, best_right_idx = None, None

    for feat in range(n_features):
        # candidate thresholds: midpoints of sorted unique values
        values = X[:, feat]
        uniq = np.unique(values)
        if len(uniq) <= 1:
            continue
        thr_candidates = (uniq[:-1] + uniq[1:]) / 2.0
        for thr in thr_candidates:
            left_mask = values <= thr
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            g_left = gini_impurity(y[left_mask])
            g_right = gini_impurity(y[right_mask])
            w_left = np.sum(left_mask) / n_samples
            w_right = 1.0 - w_left
            gain = base_gini - (w_left * g_left + w_right * g_right)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thr = thr
                best_left_idx = left_mask
                best_right_idx = right_mask
    return best_feat, best_thr, best_left_idx, best_right_idx, best_gain

def build_tree(X, y, depth=0, max_depth=50, min_samples_split=2):
    # Stopping cases
    labels, counts = np.unique(y, return_counts=True)
    majority = labels[np.argmax(counts)]
    if depth >= max_depth or len(labels) == 1 or len(y) < min_samples_split:
        return TreeNode(is_leaf=True, prediction=int(majority))

    feat, thr, left_idx, right_idx, gain = best_split(X, y)
    if feat is None or gain <= 0.0:
        return TreeNode(is_leaf=True, prediction=int(majority))

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth, min_samples_split)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth, min_samples_split)
    node = TreeNode(is_leaf=False, feature=int(feat), threshold=float(thr), left=left, right=right)
    return node

def predict_one(node, x):
    while not node.is_leaf:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.prediction

def predict(node, X):
    return np.array([predict_one(node, x) for x in X])

def load_features(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values.astype(int)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="results_dt_numpy")
    ap.add_argument("--depths", type=int, nargs="+", default=[50])
    ap.add_argument("--min_samples_split", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    Xtr, ytr = load_features(args.train)
    Xte, yte = load_features(args.test)

    all_results = {}
    for d in args.depths:
        tree = build_tree(Xtr, ytr, max_depth=d, min_samples_split=args.min_samples_split)
        yhat = predict(tree, Xte)
        metrics = compute_metrics(yte, yhat)
        cm_path = os.path.join(args.outdir, f"cm_depth{d}.png")
        plot_confusion(yte, yhat, cm_path)
        out_json = os.path.join(args.outdir, f"metrics_depth{d}.json")
        save_metrics_json(metrics, out_json)
        all_results[f"depth_{d}"] = {"metrics_json": out_json, "confusion_png": cm_path}

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Done. Results in", args.outdir)

if __name__ == "__main__":
    main()
