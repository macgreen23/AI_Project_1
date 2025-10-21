import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from icecream import ic
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from preprocessing import collect_torch_dataset

# Hyperparameters
TRAIN_SIZE = 0.8
EPOCHS = 150
BATCH_SIZE = 128
LR = 1e-2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class ConvNet(nn.Module):
    """Simple configurable CNN base class.

    Example architecture (default):
      Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)
      Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
      Flatten -> FC(hidden) -> ReLU -> FC(num_classes)

    Inputs are expected in shape (N, H, W) or (N, C, H, W). If single-channel images
    are supplied as (N, H, W) the forward pass will unsqueeze a channel dim.
    """

    def __init__(
        self, in_channels: int = 1, num_classes: int = 10, hidden_dim: int = 128
    ):
        super().__init__()
        # conv layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # adaptive pooling to fixed feature map size, then linear layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        conv_output_features = 64 * 7 * 7

        self.fc1 = nn.Linear(conv_output_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (N, H, W) or (N, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_results_dir(base: str = "results") -> Path:
    path = Path(base) / "CNN" / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_cnn(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs=10,
    batch_size=64,
    lr=1e-2,
    weight_decay=0.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    n = X.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[idx].to(device).float()
            yb = y[idx].to(device).long()

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * xb.size(0)

        epoch_loss /= n
        print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.6f}")

    return model


def evaluate_and_save(
    model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, prefix: str = "cnn"
):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        Xt = X_test.to(device).float()
        outputs = model(Xt)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    cr = classification_report(y_true, preds)
    cm = confusion_matrix(y_true, preds)

    # save
    output_dir = create_results_dir()

    # save classification report
    cr_path = output_dir / f"{prefix}_classification_report.txt"
    cr_path.write_text(cr)
    print(f"\nClassification Report:\n{cr}")
    print(f"Saved classification report to {cr_path}")

    # save confusion matrix as CSV
    cm_path = output_dir / f"{prefix}_confusion_matrix.csv"
    with cm_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in cm:
            writer.writerow(row.tolist())
    print(f"Saved confusion matrix to {cm_path}")

    # Save normalized confusion matrix visualization (rows normalized)
    try:
        cm_arr = np.array(cm)
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm_arr.astype(float) / row_sums
            cm_norm[np.isnan(cm_norm)] = 0.0

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ticks = np.arange(cm_arr.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # Annotate
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=6
                )

        plot_path = output_dir / f"{prefix}_confusion_matrix_normalized.png"
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved normalized confusion matrix to {plot_path}")
    except Exception as e:
        print(f"Could not save normalized confusion matrix plot: {e}")


def main():
    train_data, train_labels, test_data, test_labels = collect_torch_dataset(
        train_size=TRAIN_SIZE
    )

    model = ConvNet(in_channels=1, num_classes=10, hidden_dim=128)
    trained_model = train_cnn(
        model,
        train_data,
        train_labels,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
    )

    evaluate_and_save(trained_model, test_data, test_labels)


if __name__ == "__main__":
    main()
