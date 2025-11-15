
"""
VGG11 (with BatchNorm & Dropout) for CIFAR-10 (32x32 images).
Implements the architecture specified in the assignment. Trains/evaluates/saves model.

Usage examples:
  # train for 30 epochs
  python vgg11_cnn.py --mode train --epochs 30 --batch_size 128 --lr 0.01 --outdir results_vgg11

  # evaluate saved model
  python vgg11_cnn.py --mode eval --checkpoint models/vgg11_main.pth --outdir results_vgg11_eval
"""
import os, argparse, json, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from eval_utils import compute_metrics, plot_confusion, save_metrics_json, CIFAR10_CLASSES

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class VGG11(nn.Module):
    # Based on assignment spec for CIFAR-10 (32x32). After 5 MaxPools, spatial size becomes 1x1.
    def __init__(self, num_classes=10, kernel_size=3):
        super().__init__()
        k = kernel_size
        p = k // 2  # keep spatial size
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, k, 1, p), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, k, 1, p), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, k, 1, p), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, k, 1, p), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, k, 1, p), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, k, 1, p), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(512, 512, k, 1, p), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, k, 1, p), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # N x 512
        x = self.classifier(x)
        return x

def get_loaders(data_dir, batch_size):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf_train)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        preds = out.argmax(dim=1).cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","eval"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--kernel_size", type=int, default=3, help="Try 2/3/5/7 to study kernel size effects")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outdir", default="results_vgg11")
    ap.add_argument("--checkpoint", default="models/vgg11_main.pth")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    device = get_device()
    print("Device:", device)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    model = VGG11(num_classes=10, kernel_size=args.kernel_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.mode == "train":
        best_acc = 0.0
        for ep in range(1, args.epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
            y_true, y_pred = evaluate(model, test_loader, device)
            metrics = compute_metrics(y_true, y_pred)
            print(f"Epoch {ep}/{args.epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} test_acc={metrics['accuracy']:.4f}")
            # Save best
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                torch.save({"model_state": model.state_dict(),
                            "kernel_size": args.kernel_size}, args.checkpoint)
        print("Best test acc:", best_acc)

        # Final eval & artifacts
        y_true, y_pred = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        cm_path = os.path.join(args.outdir, f"cm_vgg11_k{args.kernel_size}.png")
        plot_confusion(y_true, y_pred, cm_path)
        out_json = os.path.join(args.outdir, f"metrics_vgg11_k{args.kernel_size}.json")
        save_metrics_json(metrics, out_json)

    else:  # eval
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        y_true, y_pred = evaluate(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        cm_path = os.path.join(args.outdir, f"cm_vgg11_eval.png")
        plot_confusion(y_true, y_pred, cm_path)
        out_json = os.path.join(args.outdir, "metrics_vgg11_eval.json")
        save_metrics_json(metrics, out_json)
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
