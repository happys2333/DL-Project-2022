import numpy as np
import torch
import torch.nn as nn
from torchstat import stat
from thop import profile, clever_format


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


def model_stat():
    model = nn.Conv2d(3, 32, kernel_size=3)
    a = torch.zeros((3, 64, 64))
    flops, params = profile(model, inputs=(a,))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops, params)
    print(stat(model, (3, 100, 100)))


if __name__ == "__main__":
    model_stat()
