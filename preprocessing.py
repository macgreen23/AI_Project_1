from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.8


def collect_np_dataset(train_size=TRAIN_SIZE):
    # Collect Dataset
    dataset = []
    data_dir = Path("MNIST")
    for dir in data_dir.iterdir():
        label = dir.name
        datafiles = dir.glob("*.png")
        data = [[np.asarray(Image.open(f)).flatten() / 255, label] for f in datafiles]

        dataset.extend(data)

    data, labels = zip(*dataset)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, train_size=train_size
    )

    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int32)

    return train_data, train_labels, test_data, test_labels


def collect_torch_dataset(train_size=TRAIN_SIZE):

    # Collect Dataset
    dataset = []
    data_dir = Path("MNIST")
    for dir in data_dir.iterdir():
        label = dir.name
        datafiles = dir.glob("*.png")
        data = [
            [
                F.normalize(
                    torch.tensor(np.asarray(Image.open(f)), dtype=torch.float32)
                ),
                int(label),
            ]
            for f in datafiles
        ]

        dataset.extend(data)

    data, labels = zip(*dataset)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, train_size=train_size
    )

    train_data = torch.stack(train_data)
    train_labels = torch.tensor(train_labels, dtype=torch.int64)
    test_data = torch.stack(test_data)
    test_labels = torch.tensor(test_labels, dtype=torch.int64)

    return train_data, train_labels, test_data, test_labels
