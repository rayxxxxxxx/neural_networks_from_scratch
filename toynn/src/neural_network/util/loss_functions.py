import numpy as np


def mse(y: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.linalg.norm(y_pred-y, axis=1))


def corss_entropy(P: np.ndarray, Q: np.ndarray):
    return -np.mean(np.mean(P*np.log(Q), axis=1))


def binary_corss_entropy(P: np.ndarray, Q: np.ndarray):
    return -np.mean(np.sum(P*np.log(Q)+(1-Q)*np.log(1-Q), axis=1))
