from pathlib import Path
import numpy as np

from src.neural_network.util.loss_functions import corss_entropy
from src.neural_network.model.multi_layer_perceptron import MLP, MLPOptimizer


def main():

    x_train, y_train, x_test, y_test = load_mnist_dataset()

    n_in = 784
    n_h = 32
    n_out = 10

    learning_rate = 1e-1
    batch_size = 32
    max_epoch = 4

    model = MLP(n_in, n_h, n_out)
    optimizer = MLPOptimizer(model)

    y_pred = np.array([model.forward(x) for x in x_test])
    print(f"untrained loss: {round(corss_entropy(y_test, y_pred), 3)}")

    optimizer.optimize(x_train, y_train, learning_rate, batch_size, max_epoch)

    y_pred = np.array([model.forward(x) for x in x_test])
    print(f"trained loss: {round(corss_entropy(y_test, y_pred), 3)}")

    true_positive_count = 0
    for x, y in zip(x_test, y_test):
        ypred = model.forward(x)
        true_positive_count += 1 if np.argmax(y) == np.argmax(ypred) else 0

    accuracy = true_positive_count / x_test.shape[0]
    print(f"test set accuracy: {round(accuracy*100, 2)}%")


def load_mnist_dataset():
    train_data = np.loadtxt(Path('D:/Development/Data/datasets/csv/mnist_train_small.csv'), delimiter=',')
    test_data = np.loadtxt(Path('D:/Development/Data/datasets/csv/mnist_test.csv'), delimiter=',')

    def one_hot(n_classes: int, idx: int) -> np.ndarray:
        encoding = np.zeros(n_classes)
        encoding[idx] = 1.0
        return encoding
    
    x_train = train_data[:,1:] / 255.0
    y_train = np.array([one_hot(10, int(i)) for i in train_data[:,0]])

    x_test = test_data[:,1:] / 255.0
    y_test = np.array([one_hot(10, int(i)) for i in test_data[:,0]])

    return (x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
