from pathlib import Path
import numpy as np

from src.neural_network.util.activation_functions import softmax
from src.neural_network.model.modelbase import NNModel, NNOptimizer


def calculate_grads(parameters: tuple[np.ndarray], X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray]:
    n = len(X)
    W, b = parameters
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(len(X)):
        y = softmax(np.dot(W, X[i]) + b)
        dW += np.outer(y - Y[i], X[i])
        db += y - Y[i]

    return (dW/n, db/n)


class Perceptron(NNModel):
    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.W: np.ndarray = np.random.uniform(-1, 1, (n_out, n_in))
        self.b: np.ndarray = np.zeros(n_out)

    @property
    def parameters(self) -> tuple[np.ndarray]:
        return (self.W, self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return softmax(np.dot(self.W, x)+self.b)
    
    def save(self, fp: Path) -> None:
        np.savez(fp, w=self.W, b=self.b)

    def load(self, fp: Path) -> None:
        parameters = np.load(fp)
        self.W = parameters['w']
        self.b = parameters['b']


class PerceptronOptimizer(NNOptimizer):
    def __init__(self, model: Perceptron) -> None:
        super().__init__(model)

    def calculate_grads(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray]:
        return calculate_grads(self.model.parameters, X, Y)

    def update_parameters(self, grads: tuple[np.ndarray], lr: float):
        dW, db = grads
        self.model.W -= lr * dW
        self.model.b -= lr * db
