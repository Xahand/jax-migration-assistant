import numpy as np


def normalize_and_score(x, weights, bias):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-6
    normalized = (x - mean) / std
    return normalized @ weights + bias
