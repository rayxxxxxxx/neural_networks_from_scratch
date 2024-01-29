from pathlib import Path
import numpy as np
from numba import njit, prange

from src.neural_network.util.activation_functions import relu, dRelu, softmax
from src.neural_network.model.modelbase import NNModel, NNOptimizer, NNLoader


@njit(fastmath=True)
def calc_grads(xbatch: np.ndarray, ybatch: np.ndarray, wh: np.ndarray, w: np.ndarray, bh: np.ndarray, b: np.ndarray) -> tuple[np.ndarray]:
    dwh = np.zeros(wh.shape)
    dw = np.zeros(w.shape)

    dbh = np.zeros(bh.shape)
    db = np.zeros(b.shape)

    for i in prange(xbatch.shape[0]):
        Uh = wh @ xbatch[i] + bh
        h = relu(Uh)
        y = softmax(w @ h + b)

        dw += np.outer(y - ybatch[i], h)
        db += y - ybatch[i]

        dwh += np.outer(w.T @ (y - ybatch[i]) * dRelu(Uh), xbatch[i])
        dbh += w.T @ (y - ybatch[i]) * dRelu(Uh)

    return (dwh, dw, dbh, db)


class HiddenLayerPerceptron(NNModel):
    def __init__(self, nin: int, nh: int, nout: int) -> None:
        self.nin: int = nin
        self.nh: int = nh
        self.nout: int = nout

        self.wh: np.ndarray = np.random.uniform(-1, 1, (nh, nin))
        self.bh: np.ndarray = np.zeros(nh)

        self.w: np.ndarray = np.random.uniform(-1, 1, (nout, nh))
        self.b: np.ndarray = np.zeros(nout)

    def predict(self, x: np.ndarray):
        h = relu(self.wh @ x + self.bh)
        return softmax(self.w @ h + self.b)


class HiddenLayerPerceptronOptimizer(NNOptimizer):
    def __init__(self, model: NNModel) -> None:
        super().__init__(model)

    def calc_grads(self, xbatch: np.ndarray, ybatch: np.ndarray) -> tuple[np.ndarray]:
        dwh, dw, dbh, db = calc_grads(
            xbatch,
            ybatch,
            self.model.wh,
            self.model.w,
            self.model.bh,
            self.model.b
        )

        return (dwh, dbh, dw, db)

    def apply_grads(self, grads: tuple[np.ndarray], lr: float):
        self.model.wh -= lr*grads[0]
        self.model.bh -= lr*grads[1]
        self.model.w -= lr*grads[2]
        self.model.b -= lr*grads[3]


class HiddenLayerPerceptronLoader(NNLoader):
    def __init__(self) -> None:
        super().__init__()

    def save(self, model: NNModel, dirpath: Path):
        wh_fp = Path(dirpath, 'weight_hidden.npy')
        bh_fp = Path(dirpath, 'bias_hidden.npy')

        w_fp = Path(dirpath, 'weight.npy')
        b_fp = Path(dirpath, 'bias.npy')

        np.save(wh_fp, model.wh)
        np.save(bh_fp, model.bh)

        np.save(w_fp, model.w)
        np.save(b_fp, model.b)

    def load(self, model: NNModel, dirpath: Path):
        wh_fp = Path(dirpath, 'weight_hidden.npy')
        bh_fp = Path(dirpath, 'bias_hidden.npy')

        w_fp = Path(dirpath, 'weight.npy')
        b_fp = Path(dirpath, 'bias.npy')

        model.wh = np.load(wh_fp)
        model.bh = np.load(bh_fp)

        model.w = np.load(w_fp)
        model.b = np.load(b_fp)
