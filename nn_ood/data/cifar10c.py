"""
implements dataloader for CIFAR10C
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class CIFAR10C(Dataset):
    """
    Accesses data from CIFAR10C, various corruptions applied to CIFAR10 data, as detailed in
    TODO: cite
    """

    def __init__(
        self, root, corruption="motion_blur", level=None, transform=torch.nn.Sequential(),
    ) -> None:
        super().__init__()
        DATASET_PATH = os.path.expanduser(root)
        self.corruptions = [
            "".join(path.split(".")[:-1])
            for path in os.listdir(DATASET_PATH)
            if path != "labels.npy"
        ]

        filename = os.path.join(DATASET_PATH, corruption + ".npy")
        if not os.path.exists(filename):
            raise ValueError(
                "%s is not a valid corruption, must be one of" + str(CORRUPTIONS)
            )
        if level is not None:
            if not 0 <= level < 5:
                raise ValueError("level must be None or one of [0,1,2,3,4].")

        data = np.load(filename)
        labels = np.load(os.path.join(DATASET_PATH, "labels.npy"))

        if level is not None:
            n_per_chunk = data.shape[0] // 5
            inputs = data[level * n_per_chunk : (level + 1) * n_per_chunk, ...]
            labels = torch.from_numpy(
                labels[level * n_per_chunk : (level + 1) * n_per_chunk]
            ).long()
        else:
            inputs = data
            labels = torch.from_numpy(labels).long()

        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, i):
        x = self.transform(self.inputs[i])
        y = self.labels[i]
        return x, y

    def __len__(self):
        return len(self.labels)
