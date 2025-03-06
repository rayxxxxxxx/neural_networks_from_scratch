from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np


class NNModel(ABC):    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def save(self, fp: Path) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def load(self, fp: Path) -> None:
        raise NotImplementedError()


class NNOptimizer(ABC):
    def __init__(self, model: NNModel) -> None:
        super().__init__()
        self.model: NNModel = model

    @abstractmethod
    def calculate_grads(self, xbatch: np.ndarray, ybatch: np.ndarray) -> tuple[np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def update_parameters(self, grads: tuple[np.ndarray], lr: float):
        raise NotImplementedError()

    def optimize(self, X: np.ndarray, Y: np.ndarray, lr: float, batch_size: int, max_epoch: int):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size

        for epoch in range(max_epoch):
            idxs = np.random.permutation(n_samples)
            batches = np.array_split(idxs, n_batches)

            for batch in batches:
                grads = self.calculate_grads(X[batch], Y[batch])
                self.update_parameters(grads, lr)
