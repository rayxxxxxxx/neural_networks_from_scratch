from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_network.util.loss_functions import logloss
from src.neural_network.model.perceptron import Perceptron, PerceptronOptimizer


def main():

    xtrain, ytrain, xtest, ytest = load_iris_dataset()

    nin = 4
    nout = 3

    learning_rate = 1e-2
    batch_size = 8
    max_epoch = 2000

    model = Perceptron(nin, nout)
    optimizer = PerceptronOptimizer(model)

    ypred = np.array([model.predict(x) for x in xtest])
    print('untrained loss: ', logloss(ypred, ytest))

    optimizer.optimize(
        xtrain,
        ytrain,
        learning_rate,
        batch_size,
        max_epoch
    )

    ypred = np.array([model.predict(x) for x in xtest])
    print('trained loss: ', logloss(ypred, ytest))

    true_positive_count = 0
    for x, y in zip(xtest, ytest):
        ypred = model.predict(x)
        true_positive_count += 1 if np.argmax(y) == np.argmax(ypred) else 0

    accuracy = true_positive_count / xtest.shape[0]
    print(f"test set accuracy: {round(accuracy*100, 2)}%")


def load_iris_dataset():
    df = pd.read_csv(Path('data', 'iris_csv.csv'))

    for c in df.columns[0:4]:
        df[c] = (df[c]-df[c].mean())/df[c].std()

    for name in df['class'].unique():
        df[f'label-{name}'] = df['class'].map(lambda x: 1 if x == name else 0)

    setosa_idxs = np.arange(0, 50)
    versicolor_idxs = np.arange(50, 100)
    virginica_idxs = np.arange(100, 150)

    p = np.random.permutation(np.arange(50))

    setosa_train_idxs = setosa_idxs[p[0:10]]
    setosa_test_idxs = setosa_idxs[p[10:]]

    versicolor_train_idxs = versicolor_idxs[p[0:10]]
    versicolor_test_idxs = versicolor_idxs[p[10:]]

    virginica_train_idxs = virginica_idxs[p[0:10]]
    virginica_test_idxs = virginica_idxs[p[10:]]

    feature_columns = [
        'sepallength',
        'sepalwidth',
        'petallength',
        'petalwidth'
    ]

    label_columns = [
        'label-Iris-setosa',
        'label-Iris-versicolor',
        'label-Iris-virginica'
    ]

    xtrain = np.vstack([
        df.iloc[setosa_train_idxs][feature_columns],
        df.iloc[versicolor_train_idxs][feature_columns],
        df.iloc[virginica_train_idxs][feature_columns]
    ])

    ytrain = np.vstack([
        df.iloc[setosa_train_idxs][label_columns],
        df.iloc[versicolor_train_idxs][label_columns],
        df.iloc[virginica_train_idxs][label_columns]
    ])

    xtest = np.vstack([
        df.iloc[setosa_test_idxs][feature_columns],
        df.iloc[versicolor_test_idxs][feature_columns],
        df.iloc[virginica_test_idxs][feature_columns]
    ])

    ytest = np.vstack([
        df.iloc[setosa_test_idxs][label_columns],
        df.iloc[versicolor_test_idxs][label_columns],
        df.iloc[virginica_test_idxs][label_columns]
    ])

    return (xtrain, ytrain, xtest, ytest)


if __name__ == '__main__':
    main()
