import csv
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from preprocessing import collect_np_dataset

TRAIN_SIZE = 0.80

# module-level globals used by the worker processes
_KNN_TRAIN_DATA = None
_KNN_TRAIN_LABELS = None
_KNN_K = None


def knn_worker(x):
    """Top-level worker function required for pickling by multiprocessing.

    It reads training data, labels, and k from module-level globals which are
    set in the main process before the ProcessPoolExecutor is started.
    """
    return get_knn(x, _KNN_TRAIN_DATA, _KNN_TRAIN_LABELS, _KNN_K)


def create_results_dir(base: str = "results") -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = Path(base) / "KNN" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_knn(input, train_data, train_labels, k):
    distances = np.linalg.norm(train_data - input, axis=1)
    top_k_indices = np.argsort(distances)[:k]
    top_k_labels = np.array([train_labels[i] for i in top_k_indices])
    # Count occurrences of each label
    labels, counts = np.unique(top_k_labels, return_counts=True)
    max_count = np.max(counts)
    # Find all labels with the max count
    max_labels = labels[counts == max_count]
    tie = len(max_labels) > 1
    if tie:
        return get_knn(input, train_data, train_labels, k + 1)
    return max_labels[0]


def evaluate_and_save(
    train_data, train_labels, test_data, test_labels, k, save_dir: Path, prefix="knn"
):
    num_cores = multiprocessing.cpu_count()
    global _KNN_TRAIN_DATA, _KNN_TRAIN_LABELS, _KNN_K
    _KNN_TRAIN_DATA = train_data
    _KNN_TRAIN_LABELS = train_labels
    _KNN_K = k

    with ProcessPoolExecutor(max_workers=num_cores - 2) as executor:
        predicted_labels = list(
            tqdm(
                executor.map(knn_worker, test_data),
                desc="Classifying",
                total=len(test_data),
            )
        )

    y_true = test_labels

    cr = classification_report(y_true, predicted_labels)
    cm = confusion_matrix(y_true, predicted_labels)

    print(f"\nClassification Report for k={k}:\n{cr}")

    # save classification report
    cr_path = save_dir / f"{prefix}_classification_report_k{k}.txt"
    cr_path.write_text(cr)
    print(f"Saved classification report to {cr_path}")

    # save confusion matrix as CSV
    cm_path = save_dir / f"{prefix}_confusion_matrix_k{k}.csv"
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

        plot_path = save_dir / f"{prefix}_confusion_matrix_normalized_k{k}.png"
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved normalized confusion matrix to {plot_path}")
    except Exception as e:
        print(f"Could not save normalized confusion matrix plot: {e}")


def main():
    save_dir = create_results_dir()
    train_data, train_labels, test_data, test_labels = collect_np_dataset(TRAIN_SIZE)
    for k in tqdm([1, 3, 5]):
        evaluate_and_save(train_data, train_labels, test_data, test_labels, k, save_dir)


if __name__ == "__main__":
    main()
