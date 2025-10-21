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

# Hyperparameters
TRAIN_SIZE = 0.8
EPOCHS = 150
BATCH_SIZE = 128
LR = 1e-2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int):
        super(MultilayerPerceptron, self).__init__()
        self.mlp = self._build_network(input_size, hidden_sizes, num_classes)
        self.num_classes = num_classes

    def _build_network(
        self, input_size: int, hidden_sizes: list[int], output_size: int
    ):
        layers = []
        in_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def create_results_dir(base: str = "results") -> Path:
    path = Path(base) / "MLP" / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def train(
    model: MultilayerPerceptron,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs=10,
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
            yb_onehot = F.one_hot(yb, num_classes=model.mlp[-1].out_features).float()

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
    model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor, prefix: str = "mlp"
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
    print(f"Saved classification report to {cr_path}")
    print(f"\nClassification Report for MLP:\n{cr}")

    # save confusion matrix as CSV
    cm_path = out_dir / f"{prefix}_confusion_matrix.csv"
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

        plot_path = out_dir / f"{prefix}_confusion_matrix_normalized.png"
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved normalized confusion matrix to {plot_path}")
    except Exception as e:
        print(f"Could not save normalized confusion matrix plot: {e}")


def main():
    # Hyperparameters
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    lr = LR

    train_data, train_labels, test_data, test_labels = collect_torch_dataset(TRAIN_SIZE)

    # Flatten inputs: collect_torch_dataset returns tensors shaped (N, H, W) or (N, H, W, C)
    X_train = train_data.view(train_data.shape[0], -1)
    X_test = test_data.view(test_data.shape[0], -1)

    model = MultilayerPerceptron(784, [256, 128], 10)

    model = train(
        model, X_train, train_labels, epochs=epochs, batch_size=batch_size, lr=lr
    )

    evaluate_and_save(model, X_test, test_labels, prefix="mlp")


if __name__ == "__main__":
    main()
