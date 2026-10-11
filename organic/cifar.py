"""CIFAR-10 (Krizhevsky 2009): 60000 small photos in 10 classes.

Labels are used only to *read* what the organ learned (nearest-neighbour
tests), never to teach it. Coarse grouping: animals vs vehicles.
"""

import os
import pickle

import numpy as np

ROOT = os.path.expanduser(os.environ.get("CIFAR_DIR", "~/datasets/cifar-10-batches-py"))
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
ANIMAL = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])


def _batch(name):
    with open(os.path.join(ROOT, name), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    return x, np.array(d[b"labels"])


def load(split="train"):
    if split == "train":
        parts = [_batch(f"data_batch_{i}") for i in range(1, 6)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _batch("test_batch")
