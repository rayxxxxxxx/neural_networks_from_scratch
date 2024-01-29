import numpy as np
from numba import njit


@njit(fastmath=True)
def ident(x: np.ndarray) -> np.ndarray:
    return x


@njit(fastmath=True)
def dIdent(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


@njit(fastmath=True)
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.zeros(x.shape), x)


@njit(fastmath=True)
def dRelu(x: np.ndarray) -> np.ndarray:
    return 1 * (x > 0)


@njit(fastmath=True)
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


@njit(fastmath=True)
def dtanh(x: np.ndarray) -> np.ndarray:
    return 1-np.square(np.tanh(x))


@njit(fastmath=True)
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))


@njit(fastmath=True)
def dSigmoid(x: np.ndarray) -> np.ndarray:
    y = 1/(1+np.exp(-x))
    return y*(1-y)


@njit(fastmath=True)
def softmax(x: np.ndarray) -> np.ndarray:
    y = np.exp(x)
    return y/np.sum(y)
