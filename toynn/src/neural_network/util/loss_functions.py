import numpy as np
from src.neural_network.model.modelbase import NNModel


def loss(model: NNModel, X: np.ndarray, Y: np.ndarray, loss_fn: object) -> float:
    y_pred = np.array([model.forward(x) for x in X])
    return loss_fn(Y, y_pred)


def mse(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.linalg.norm(y_pred-y, axis=1))


def corss_entropy(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return -np.mean(np.mean(P*np.log(Q), axis=1))


def binary_corss_entropy(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return -np.mean(np.sum(P*np.log(Q)+(1-Q)*np.log(1-Q), axis=1))
