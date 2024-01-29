from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np


class NNModel(ABC):
    @abstractmethod
    def predict(self, x: np.ndarray):
        raise NotImplementedError()


class NNOptimizer(ABC):
    def __init__(self, model: NNModel) -> None:
        super().__init__()
        self.model = model

    @abstractmethod
    def calc_grads(self, xbatch: np.ndarray, ybatch: np.ndarray) -> tuple[np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def apply_grads(self, grads: tuple[np.ndarray], lr: float):
        raise NotImplementedError()

    def optimize(self, xtrain: np.ndarray, ytrain: np.ndarray, lr: float, batch_size: int, max_epoch: int):
        n_samples = xtrain.shape[0]

        for epoch in range(max_epoch):
            idxs = np.random.permutation(n_samples)

            for i in range(n_samples//batch_size):
                ibegin = i * batch_size
                iend = min((i + 1) * batch_size, n_samples - 1)
                batch_idxs = idxs[ibegin:iend]

                grads = self.calc_grads(xtrain[batch_idxs], ytrain[batch_idxs])
                self.apply_grads(grads, lr)


class NNLoader(ABC):
    @abstractmethod
    def save(self, model: NNModel, dirpath: Path):
        raise NotImplementedError()

    @abstractmethod
    def load(self, model: NNModel, dirpath: Path):
        raise NotImplementedError()
