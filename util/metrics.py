import numpy as np


def check(x: np.ndarray, y: np.ndarray):
    assert x.shape == y.shape
    if x.ndim == 1:
        x, y = x.reshape(1, x.shape[0]), y.reshape(1, x.shape[0])
    return x, y


def MSE(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = check(x, y)
    N, L = x.shape
    mse = np.sum((x - y) ** 2, axis=1) / L

    return mse


def MAE(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x, y = check(x, y)
    N, L = x.shape
    mae = np.sum(np.abs(x - y), axis=1) / L

    return mae


if __name__ == "__main__":
    x = np.array(((1, 2, 3), (3, 2, 1)))
    y = np.array(((1, 2, 3), (2, 2, 2)))
    mse = MSE(x, y)
    print(mse)
