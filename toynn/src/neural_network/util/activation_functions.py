import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return x * (x > 0)


def drelu(x: np.ndarray) -> np.ndarray:
    return 1.0 * (x > 0)


def dtanh(x: np.ndarray) -> np.ndarray:
    return 1 - np.square(np.tanh(x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))


def dsigmoid(x: np.ndarray) -> np.ndarray:
    y = 1 / (1 + np.exp(-x))
    return y * (1 - y)


def softmax(x: np.ndarray) -> np.ndarray:
    y = np.exp(x)
    return y / np.sum(y)
