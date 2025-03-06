import numpy as np
from src.neural_network.model.modelbase import NNModel

def accuracy(model: NNModel, X: np.ndarray, Y: np.ndarray) -> float:
    true_positive_count = 0
    for x, y in zip(X, Y):
        y_pred = model.forward(x)
        true_positive_count += 1 if np.argmax(y) == np.argmax(y_pred) else 0
    return round(true_positive_count / len(X) * 100, 2)