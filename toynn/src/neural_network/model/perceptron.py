from pathlib import Path
import numpy as np
from numba import njit, prange

from src.neural_network.util.activation_functions import softmax
from src.neural_network.model.modelbase import NNModel, NNOptimizer, NNLoader


@njit(fastmath=True)
def calc_grads(xbatch: np.ndarray, ybatch: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray]:
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    for i in prange(xbatch.shape[0]):
        y = softmax(w @ xbatch[i] + b)
        dw += np.outer(y - ybatch[i], xbatch[i])
        db += y - ybatch[i]

    return dw, db


class Perceptron(NNModel):
    def __init__(self, nin: int, nout: int) -> None:
        super().__init__()
        self.nin: int = nin
        self.nout: int = nout
        self.w: np.ndarray = np.random.uniform(-1, 1, (nout, nin))
        self.b: np.ndarray = np.zeros(nout)

    def predict(self, x: np.ndarray):
        return softmax(self.w @ x + self.b)


class PerceptronOptimizer(NNOptimizer):
    def __init__(self, model: NNModel) -> None:
        super().__init__(model)

    def calc_grads(self, xbatch: np.ndarray, ybatch: np.ndarray) -> tuple[np.ndarray]:
        dw, db = calc_grads(
            xbatch,
            ybatch,
            self.model.w,
            self.model.b
        )

        return (dw, db)

    def apply_grads(self, grads: tuple[np.ndarray], lr: float):
        self.model.w -= lr*grads[0]
        self.model.b -= lr*grads[1]


class PerceptronLoader(NNLoader):
    def __init__(self) -> None:
        super().__init__()

    def save(self, model: NNModel, dirpath: Path):
        w_fp = Path(dirpath, 'weight.npy')
        b_fp = Path(dirpath, 'bias.npy')

        np.save(w_fp, model.w)
        np.save(b_fp, model.b)

    def load(self, model: NNModel, dirpath: Path):
        w_fp = Path(dirpath, 'weight.npy')
        b_fp = Path(dirpath, 'bias.npy')

        model.w = np.load(w_fp)
        model.b = np.load(b_fp)
