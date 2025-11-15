import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ===== Device Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== Load CSV features =====
train_df = pd.read_csv("features/train_features.csv")
test_df = pd.read_csv("features/test_features.csv")

X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values

X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

# ===== Define the MLP =====
class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.model(x)

# ===== Define MLP Variants by Depth =====
class MLP_1Layer(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.model(x)

class MLP_3Layer(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.model(x)


# ===== Evaluation Function =====
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, digits=4))

    # Plot confusion matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return accuracy, classification_report(all_labels, all_preds, output_dict=True)

# ===== Class Names =====
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]


# ===== Evaluate the saved base model =====
print("\n===== Evaluating mlp_base =====")
base_model = MLP(hidden_size=512).to(device)
base_model.load_state_dict(torch.load("mlp_base.pth", map_location=device))
acc_base, report_base = evaluate_model(base_model, test_loader, class_names)

# ===== Depth Variants Dictionary =====
depth_variants = {
    "mlp_1layer": MLP_1Layer,
    "mlp_3layer": MLP_3Layer
}

summary_depth = []

import torch.optim as optim
criterion = nn.CrossEntropyLoss()

# ===== Train and Evaluate Depth Variants =====
for name, ModelClass in depth_variants.items():
    print(f"\n===== Training {name} =====")
    model = ModelClass().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(30):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), f"{name}.pth")
    print(f"{name} saved.")

    acc, report = evaluate_model(model, test_loader, class_names)
    summary_depth.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": np.mean([report[str(i)]["precision"] for i in range(10)]),
        "Recall": np.mean([report[str(i)]["recall"] for i in range(10)]),
        "F1": np.mean([report[str(i)]["f1-score"] for i in range(10)])
    })


# ===== Evaluate/train other variants =====
#variants = {
#    "mlp_narrow": 256,
#    "mlp_wide": 1024,
#    "mlp_extra_wide": 2048
#}

summary_metrics = [{
    "Model": "mlp_base",
    "Accuracy": acc_base,
    "Precision": np.mean([report_base[str(i)]["precision"] for i in range(10)]),
    "Recall": np.mean([report_base[str(i)]["recall"] for i in range(10)]),
    "F1": np.mean([report_base[str(i)]["f1-score"] for i in range(10)])
}]

"""
# Example: If you want to train/evaluate other variants
train_variants = True  # Set True if you want to train variants
if train_variants:
    import torch.optim as optim

    for variant_name, hidden_size in variants.items():
        print(f"\n===== Training {variant_name} =====")
        model = MLP(hidden_size=hidden_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Training loop (simplified)
        for epoch in range(30):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Save variant
        torch.save(model.state_dict(), f"{variant_name}.pth")
        print(f"{variant_name} saved.")

        # Evaluate
        acc, report = evaluate_model(model, test_loader, class_names)
        summary_metrics.append({
            "Model": variant_name,
            "Accuracy": acc,
            "Precision": np.mean([report[str(i)]["precision"] for i in range(10)]),
            "Recall": np.mean([report[str(i)]["recall"] for i in range(10)]),
            "F1": np.mean([report[str(i)]["f1-score"] for i in range(10)])
        })
"""
# ===== Summary Table =====
#summary_df = pd.DataFrame(summary_metrics)
#print("\n===== Summary Table =====")
#print(summary_df)
#summary_df.to_csv("mlp_variants_summary.csv", index=False)
#print("Summary metrics saved to mlp_variants_summary.csv")

# ===== Summary Table =====
depth_df = pd.DataFrame(summary_depth)
print("\n===== Depth Experiment Summary =====")
print(depth_df)
depth_df.to_csv("mlp_depth_summary.csv", index=False)
print("Depth experiment metrics saved to mlp_depth_summary.csv")

