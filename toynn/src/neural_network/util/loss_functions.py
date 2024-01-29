import math
import numpy as np


def mse(ypred: np.ndarray, y: np.ndarray):
    return np.mean(np.linalg.norm(ypred - y, axis=1))


def logloss(ypred: np.ndarray, y: np.ndarray):
    return -np.mean(np.mean(y * np.log(ypred), axis=1))


def binary_logloss(ypred: float, y: float):
    return -(y * math.log(ypred) + (1 - y) * math.log(1 - ypred))
