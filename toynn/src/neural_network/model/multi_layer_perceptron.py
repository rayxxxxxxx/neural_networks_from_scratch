from pathlib import Path
import numpy as np

from src.neural_network.util.activation_functions import sigmoid, dsigmoid, softmax
from src.neural_network.model.modelbase import NNModel, NNOptimizer


def calculate_grads(parameters: tuple[np.ndarray], X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray]:
    n = len(X)
    W1, b1, W2, b2 = parameters
    dW1, db1 = np.zeros(W1.shape), np.zeros(b1.shape)
    dW2, db2 = np.zeros(W2.shape), np.zeros(b2.shape)

    for i in range(len(X)):
        u = W1 @ X[i] + b1
        h = sigmoid(u)
        y = softmax(W2 @ h + b2)
        
        dLdy = y - Y[i]
        dLdu = W2.T @ dLdy * dsigmoid(u)

        dW1 += np.outer(dLdu, X[i])
        db1 += dLdu

        dW2 += np.outer(dLdy, h)
        db2 += dLdy

    return (dW1/n, db1/n, dW2/n, db2/n)


class MLP(NNModel):
    def __init__(self, n_in: int, n_h: int, n_out: int) -> None:
        super().__init__()

        self.W1: np.ndarray = np.random.uniform(-1, 1, (n_h, n_in))
        self.b1: np.ndarray = np.zeros(n_h)

        self.W2: np.ndarray = np.random.uniform(-1, 1, (n_out, n_h))
        self.b2: np.ndarray = np.zeros(n_out)

    @property
    def parameters(self) -> tuple[np.ndarray]:
        return (self.W1, self.b1, self.W2, self.b2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = sigmoid(self.W1 @ x + self.b1)
        return softmax(self.W2 @ h + self.b2)

    def save(self, fp: Path) -> None:
        np.savez(fp, w1=self.W1, b1=self.b1, w2=self.W2, b2=self.b2)

    def load(self, fp: Path) -> None:
        parameters = np.load(fp)
        self.W1, self.b1 = parameters['w1'], parameters['b1']
        self.W2, self.b2 = parameters['w2'], parameters['b2']


class MLPOptimizer(NNOptimizer):
    def __init__(self, model: NNModel) -> None:
        super().__init__(model)

    def calculate_grads(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray]:
        return calculate_grads(self.model.parameters, X, Y)

    def update_parameters(self, grads: tuple[np.ndarray], lr: float):
        dW1, db1, dW2, db2 = grads
        self.model.W1 -= lr * dW1
        self.model.b1 -= lr * db1
        self.model.W2 -= lr * dW2
        self.model.b2 -= lr * db2
