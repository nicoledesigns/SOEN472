import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.decomposition import PCA

# loads the required libraries and sets random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Chooses GPU if availale 
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# resize the images from CIFAR-10 (32x32) to 224x224 and normalize them
def build_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# iterates through the dataset and collects k samples per class for the 10 tables
# stops when each class reaches k samples
def subset_per_class(ds, k):
    buckets = defaultdict(list)
    for idx, (_, label) in enumerate(ds):
        if len(buckets[label]) < k:
            buckets[label].append(idx)
        if len(buckets) == 10 and all(len(b) == k for b in buckets.values()):
            break
    return Subset(ds, [i for c in sorted(buckets) for i in buckets[c]])

# loads pretrained ResNet-18, removes the final classification layer
# replaces final classification (fc) layer with identity function to get feature vectors
def extract_features(loader, device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.to(device).eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            feats.append(out.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)

# builds a dataframe with columns from 1 to 50
def save_csv(X, y, path):
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(1, X.shape[1]+1)])
    df["label"] = y
    df.to_csv(path, index=False)
    print("Saved", path)

# main function to orchestrate the feature extraction and saving process
def main():
    data_dir = "data"
    out_dir = "features"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()
    print("Device:", device)

    # make sure the folder exists
    tfm = build_transforms()
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)

    # build balances subsets
    train_sub = subset_per_class(train_ds, 500)
    test_sub = subset_per_class(test_ds, 100)

    # data loaders
    train_loader = DataLoader(train_sub, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_sub, batch_size=128, shuffle=False, num_workers=2)

    # call feature extraction for all selected images
    Xtr, ytr = extract_features(train_loader, device)
    Xte, yte = extract_features(test_loader, device)

    pca = PCA(n_components=50)
    Xtr_50 = pca.fit_transform(Xtr)
    Xte_50 = pca.transform(Xte)

    save_csv(Xtr_50, ytr, os.path.join(out_dir, "train_features.csv"))
    save_csv(Xte_50, yte, os.path.join(out_dir, "test_features.csv"))

if __name__ == "__main__":
    main()