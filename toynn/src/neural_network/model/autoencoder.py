from pathlib import Path
import numpy as np
from numba import njit, prange

from src.neural_network.util.activation_functions import relu, dRelu
from src.neural_network.model.modelbase import NNModel, NNOptimizer, NNLoader


@njit(fastmath=True)
def calc_grads(xbatch: np.ndarray, ybatch: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> tuple[np.ndarray]:
    dw1 = np.zeros(w1.shape)
    db1 = np.zeros(b1.shape)

    dw2 = np.zeros(w2.shape)
    db2 = np.zeros(b2.shape)

    for i in prange(xbatch.shape[0]):
        h = w1 @ xbatch[i] + b1
        z = relu(h)
        y = w2 @ z + b2

        dw2 += np.outer(2 * (y - ybatch[i]), z)
        db2 += 2 * (y - ybatch[i])

        dw1 += np.outer(w2.T @ (2 * (y - ybatch[i])) * dRelu(h), xbatch[i])
        db1 += w2.T @ (2 * (y - ybatch[i])) * dRelu(h)

    return (dw1, db1, dw2, db2)


class Autoencoder(NNModel):
    def __init__(self, nin: int, nh: int) -> None:
        self.nin: int = nin
        self.nh: int = nh

        self.w1: np.ndarray = np.random.uniform(-1, 1, (nh, nin))
        self.b1: np.ndarray = np.zeros(nh)

        self.w2: np.ndarray = np.random.uniform(-1, 1, (nin, nh))
        self.b2: np.ndarray = np.zeros(nin)

    def predict(self, x: np.ndarray):
        z = relu(self.w1 @ x + self.b1)
        return self.w2 @ z + self.b2


class AutoencoderOptimizer(NNOptimizer):
    def __init__(self, model: NNModel) -> None:
        super().__init__(model)

    def calc_grads(self, xbatch: np.ndarray, ybatch: np.ndarray) -> tuple[np.ndarray]:
        dw1, db1, dw2, db2 = calc_grads(
            xbatch,
            ybatch,
            self.model.w1,
            self.model.b1,
            self.model.w2,
            self.model.b2
        )

        return (dw1, db1, dw2, db2)

    def apply_grads(self, grads: tuple[np.ndarray], lr: float):
        self.model.w1 -= lr*grads[0]
        self.model.b1 -= lr*grads[1]
        self.model.w2 -= lr*grads[2]
        self.model.b2 -= lr*grads[3]


class AutoencoderLoader(NNLoader):
    def __init__(self) -> None:
        super().__init__()

    def save(self, model: NNModel, dirpath: Path):
        w1_fp = Path(dirpath, 'weight_1.npy')
        b1_fp = Path(dirpath, 'bias_1.npy')

        w2_fp = Path(dirpath, 'weight_2.npy')
        b2_fp = Path(dirpath, 'bias_2.npy')

        np.save(w1_fp, model.w1)
        np.save(b1_fp, model.b1)

        np.save(w2_fp, model.w2)
        np.save(b2_fp, model.b2)

    def load(self, model: NNModel, dirpath: Path):
        w1_fp = Path(dirpath, 'weight_1.npy')
        b1_fp = Path(dirpath, 'bias_1.npy')

        w2_fp = Path(dirpath, 'weight_2.npy')
        b2_fp = Path(dirpath, 'bias_2.npy')

        model.w1 = np.load(w1_fp)
        model.b1 = np.load(b1_fp)

        model.w2 = np.load(w2_fp)
        model.b2 = np.load(b2_fp)
