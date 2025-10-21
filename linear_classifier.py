import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import collect_torch_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Hyperparameters
TRAIN_SIZE = 0.8
EPOCHS = 150
BATCH_SIZE = 128
LR = 1e-2


class LinearClassifier(nn.Module):
    """Simple linear classifier y = W x + b"""

    def __init__(self, input_size: int, num_classes: int):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def create_results_dir(base: str = "results") -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = Path(base) / "LinearClassifier" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def train(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
):
    device = next(model.parameters()).device
    n = X.shape[0]
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[idx].to(device).float()
            yb = y[idx].to(device)
            # yb is integer labels; convert to one-hot for MSE
            yb_onehot = F.one_hot(yb, num_classes=model.linear.out_features).float()

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb_onehot)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * xb.size(0)

        epoch_loss /= n
        print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.6f}")

    return model


def evaluate_and_save(
    model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, prefix: str = "linear"
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

    out_dir = create_results_dir()

    # save classification report
    cr_path = out_dir / f"{prefix}_classification_report.txt"
    cr_path.write_text(cr)
    print(f"\nClassification Report for Linear Classifier:\n{cr}")

    # save confusion matrix as CSV
    cm_path = out_dir / f"{prefix}_confusion_matrix.csv"
    with cm_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cm)
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

        plot_path = out_dir / f"{prefix}_confusion_matrix_normalized.png"
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved normalized confusion matrix to {plot_path}")
    except Exception as e:
        print(f"Could not save normalized confusion matrix plot: {e}")

    # Visualize weight matrix W as digit-like images if a single linear layer exists
    try:
        linear = model.linear
        W = linear.weight.detach().cpu().numpy()  # shape (num_classes, input_size)
        num_classes, input_size = W.shape

        # Infer image shape: try square, otherwise fallback to (28,28)
        img_side = int(np.round(np.sqrt(input_size)))
        if img_side * img_side != input_size:
            img_side = 28

        # Create per-class weight images
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        for cls in range(num_classes):
            w = W[cls].reshape(img_side, img_side)
            plt.figure(figsize=(3, 3))
            plt.imshow(w, cmap="seismic", interpolation="nearest")
            plt.colorbar(shrink=0.6)
            plt.title(f"Class {cls} weights")
            plt.axis("off")
            fn = weights_dir / f"linear_weights_class_{cls}.png"
            plt.savefig(fn, bbox_inches="tight")
            plt.close()

        # Create a grid of weight images
        cols = min(10, num_classes)
        rows = int(np.ceil(num_classes / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = np.array(axes).reshape(-1)
        for i in range(rows * cols):
            ax = axes[i]
            ax.axis("off")
            if i < num_classes:
                w = W[i].reshape(img_side, img_side)
                ax.imshow(w, cmap="seismic", interpolation="nearest")
                ax.set_title(str(i), fontsize=8)

        grid_fn = out_dir / "linear_weights_grid.png"
        plt.tight_layout()
        plt.savefig(grid_fn, bbox_inches="tight")
        plt.close()
        print(f"Saved weight visualizations to {weights_dir} and {grid_fn}")
    except Exception as e:
        print(f"Could not visualize weights: {e}")


def main():
    # Hyperparameters
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    lr = LR

    train_data, train_labels, test_data, test_labels = collect_torch_dataset(TRAIN_SIZE)

    # Flatten inputs: collect_torch_dataset returns tensors shaped (N, H, W) or (N, H, W, C)
    X_train = train_data.view(train_data.shape[0], -1)
    X_test = test_data.view(test_data.shape[0], -1)

    input_size = X_train.shape[1]
    num_classes = int(torch.max(train_labels).item() + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearClassifier(input_size=input_size, num_classes=num_classes).to(device)

    model = train(
        model, X_train, train_labels, epochs=epochs, batch_size=batch_size, lr=lr
    )

    evaluate_and_save(model, X_test, test_labels)


if __name__ == "__main__":
    main()
