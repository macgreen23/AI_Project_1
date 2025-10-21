import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import collect_np_dataset

TRAIN_SIZE = 0.8


class NaiveBayes:
    def __init__(self):
        self.classes = None  # Class labels
        self.class_count = None  # Count of each class
        self.class_log_priors = None  # P(y) shape (n_classes,)
        self.feature_log_prob = None  # P(x_i=1|y) shape (n_classes, n_features)
        self.feature_log_prob_neg = None  # P(x_i=0|y) shape (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Naive Bayes model to the training data.
        param X: Training data, shape (n_samples, n_features)
        param y: Training labels, shape (n_samples,)
        """

        n_samples, n_features = X.shape
        classes, class_counts = np.unique(y, return_counts=True)
        n_classes = classes.shape[0]

        self.classes = classes
        self.class_count = class_counts
        self.class_log_priors = np.log(class_counts / n_samples)

        # Calculate feature likelihoods
        self.feature_log_prob = np.zeros((n_classes, n_features))
        self.feature_log_prob_neg = np.zeros((n_classes, n_features))

        for cls in self.classes:
            X_cls = X[y == cls]
            feature_counts = X_cls.sum(axis=0)
            self.feature_log_prob[cls] = np.log(
                (feature_counts + 1) / (X_cls.shape[0] + 2)
            )
            self.feature_log_prob_neg[cls] = np.log(
                (X_cls.shape[0] - feature_counts + 1) / (X_cls.shape[0] + 2)
            )

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the unnormalized posterior log probability of X
        param X: Input data, shape (n_samples, n_features)
        return: Joint log likelihood, shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        pos = X @ self.feature_log_prob.T
        neg = (1 - X) @ self.feature_log_prob_neg.T
        return pos + neg + self.class_log_priors

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        param X: Input data, shape (n_samples, n_features)
        return: Predicted class labels, shape (n_samples,)
        """
        jll = self._joint_log_likelihood(X)
        return self.classes[np.argmax(jll, axis=1)]


def create_results_dir(base: str = "results") -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = Path(base) / "Naive_Bayes" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_and_save(
    model: NaiveBayes, test_data: np.ndarray, test_labels: np.ndarray, save_dir: Path
):
    predicted_labels = model.predict(test_data)

    cr = classification_report(test_labels, predicted_labels)
    cm = confusion_matrix(test_labels, predicted_labels)

    print(f"\nClassification Report for Naive Bayes:\n{cr}")

    # save classification report
    cr_path = save_dir / f"naive_bayes_classification_report.txt"
    cr_path.write_text(cr)
    print(f"Saved classification report to {cr_path}")

    # save confusion matrix as CSV
    cm_path = save_dir / f"naive_bayes_confusion_matrix.csv"
    with cm_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cm)
    print(f"Saved confusion matrix to {cm_path}")

    # Visualize per-class feature probabilities as images
    try:
        probs = model.feature_log_prob.copy()  # log probabilities
        probs = np.exp(probs)  # convert to probabilities
        n_classes, n_features = probs.shape

        # Infer image side length
        img_side = int(np.round(np.sqrt(n_features)))
        if img_side * img_side != n_features:
            img_side = 28

        probs_dir = save_dir / "prob_maps"
        probs_dir.mkdir(parents=True, exist_ok=True)

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
                        j,
                        i,
                        f"{cm_norm[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )

            plot_path = save_dir / f"naive_bayes_confusion_matrix_normalized.png"
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved normalized confusion matrix to {plot_path}")
        except Exception as e:
            print(f"Could not save normalized confusion matrix plot: {e}")
        for cls in range(n_classes):
            p = probs[cls].reshape(img_side, img_side)
            plt.figure(figsize=(3, 3))
            plt.imshow(p, cmap="viridis", interpolation="nearest")
            plt.colorbar(shrink=0.6)
            plt.title(f"P(x=1|y={cls})")
            plt.axis("off")
            fn = probs_dir / f"naive_bayes_prob_class_{cls}.png"
            plt.savefig(fn, bbox_inches="tight")
            plt.close()

        # Grid of probability maps
        cols = min(10, n_classes)
        rows = int(np.ceil(n_classes / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        axes = np.array(axes).reshape(-1)
        for i in range(rows * cols):
            ax = axes[i]
            ax.axis("off")
            if i < n_classes:
                p = probs[i].reshape(img_side, img_side)
                ax.imshow(p, cmap="viridis", interpolation="nearest")
                ax.set_title(str(i), fontsize=8)

        grid_fn = save_dir / "naive_bayes_prob_grid.png"
        plt.tight_layout()
        plt.savefig(grid_fn, bbox_inches="tight")
        plt.close()
        print(
            f"Saved Naive Bayes probability visualizations to {probs_dir} and {grid_fn}"
        )
    except Exception as e:
        print(f"Could not visualize Naive Bayes probabilities: {e}")


def main():
    save_dir = create_results_dir()
    train_data, train_labels, test_data, test_labels = collect_np_dataset(TRAIN_SIZE)
    train_data = np.where(train_data > 0.5, 1, 0)
    test_data = np.where(test_data > 0.5, 1, 0)

    model = NaiveBayes()
    model.fit(train_data, train_labels)

    evaluate_and_save(model, test_data, test_labels, save_dir)


if __name__ == "__main__":
    main()
