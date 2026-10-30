import os

import numpy as np
import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def load_xcheck(tag):
    with open(os.path.join(FIX, f"xc_{tag}_dims.txt")) as fh:
        N, T = map(int, fh.read().split())
    y = np.loadtxt(os.path.join(FIX, f"xc_{tag}_y.csv"), delimiter=",")
    z = np.loadtxt(os.path.join(FIX, f"xc_{tag}_z.csv"), delimiter=",")
    zf = np.loadtxt(os.path.join(FIX, f"xc_{tag}_zf.csv"), delimiter=",")
    Y = y.reshape(N, T)
    W = np.concatenate([np.ones((N, T, 1)), z.reshape(N, T, -1)], axis=2)
    wn = np.c_[np.ones(N), zf]
    ref = np.loadtxt(os.path.join(FIX, f"xc_{tag}_matlab_forecasts.csv"), delimiter=",")
    return Y, W, wn, ref


@pytest.fixture
def small_panel():
    rng = np.random.default_rng(42)
    N, T = 30, 25
    theta = np.c_[1 + 0.5 * rng.normal(size=N), 0.5 + 0.1 * rng.normal(size=N),
                  0.3 + 0.2 * rng.normal(size=N)]
    x = rng.normal(size=(N, T + 2))
    y = np.zeros((N, T + 2))
    for t in range(1, T + 2):
        y[:, t] = theta[:, 0] + theta[:, 1] * y[:, t - 1] + theta[:, 2] * x[:, t] + rng.normal(size=N)
    W = np.stack([np.ones((N, T)), y[:, 0:T], x[:, 1:T + 1]], axis=2)
    return y[:, 1:T + 1], W, np.c_[np.ones(N), y[:, T], x[:, T + 1]], theta
