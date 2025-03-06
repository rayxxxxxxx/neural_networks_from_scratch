from pathlib import Path
import numpy as np

from src.neural_network.util.activation_functions import relu, drelu
from src.neural_network.model.modelbase import NNModel, NNOptimizer


def calculate_grads(parameters: tuple[np.ndarray], X: np.ndarray) -> tuple[np.ndarray]:
    n = len(X)
    W1, b1, W2, b2 = parameters
    dW1, db1 = np.zeros(W1.shape), np.zeros(b1.shape)
    dW2, db2 = np.zeros(W2.shape), np.zeros(b2.shape)

    for i in range(n):
        h = W1 @ X[i] + b1
        z = relu(h)
        y = W2 @ z + b2

        dLdy = 2 * (y - X[i])
        dLdz = W2.T @ dLdy
        dLdh = dLdz * drelu(h)

        dW2 += np.outer(dLdy, z)
        db2 += dLdy

        dW1 += np.outer(dLdh, X[i])
        db1 += dLdh

    return (dW1/n, db1/n, dW2/n, db2/n)


class Autoencoder(NNModel):
    def __init__(self, n_in: int, n_h: int) -> None:
        super().__init__()
        
        self.W1: np.ndarray = np.random.uniform(-1, 1, (n_h, n_in))
        self.b1: np.ndarray = np.zeros(n_h)

        self.W2: np.ndarray = np.random.uniform(-1, 1, (n_in, n_h))
        self.b2: np.ndarray = np.zeros(n_in)

    @property
    def parameters(self) -> tuple[np.ndarray]:
        return (self.W1, self.b1, self.W2, self.b2)

    def forward(self, x: np.ndarray):
        z = relu(self.W1 @ x + self.b1)
        return self.W2 @ z + self.b2
    
    def save(self, fp: Path) -> None:
        np.savez(fp, w1=self.W1, b1=self.b1, w2=self.W2, b2=self.b2)

    def load(self, fp: Path) -> None:
        param = np.load(fp)
        self.W1, self.b1 = param['w1'], param['b1']
        self.W2, self.b2 = param['w2'], param['b2']


class AutoencoderOptimizer(NNOptimizer):
    def __init__(self, model: NNModel) -> None:
        super().__init__(model)

    def calculate_grads(self, X: np.ndarray) -> tuple[np.ndarray]:
        return calculate_grads(self.model.parameters, X)

    def update_parameters(self, grads: tuple[np.ndarray], lr: float):
        self.model.W1 -= lr*grads[0]
        self.model.b1 -= lr*grads[1]
        self.model.W2 -= lr*grads[2]
        self.model.b2 -= lr*grads[3]
