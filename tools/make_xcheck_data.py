"""Simulated panels used to cross-check the Python port against the MATLAB code."""
import numpy as np, sys, os
out = sys.argv[1]
for tag, N, T, seed in [("a", 40, 21, 1), ("b", 25, 30, 2), ("c", 60, 12, 3)]:
    rng = np.random.default_rng(seed)
    beta = 0.5 + rng.uniform(-0.25, 0.25, N)
    gam = 0.3 + 0.3 * rng.normal(size=N)
    alpha = 1 + rng.normal(size=N)
    x = rng.normal(size=(N, T + 2)) + rng.normal(size=(N, 1))
    y = np.zeros((N, T + 2))
    for t in range(1, T + 2):
        y[:, t] = alpha + beta * y[:, t - 1] + gam * x[:, t] + rng.normal(size=N) * (0.5 + rng.uniform(size=N))
    # estimation: y_t on (1, y_{t-1}, x_t) for t = 2..T+1 ; forecast regressors (y_{T+1}, x_{T+2})
    Y = y[:, 1:T + 1]; ylag = y[:, 0:T]; X = x[:, 1:T + 1]
    zf = np.c_[y[:, T], x[:, T + 1]]
    stacked_y = Y.reshape(-1)            # unit-major stacking = MATLAB vec of TxN
    stacked_z = np.c_[ylag.reshape(-1), X.reshape(-1)]
    np.savetxt(os.path.join(out, f"xc_{tag}_y.csv"), stacked_y, delimiter=",", fmt="%.17g")
    np.savetxt(os.path.join(out, f"xc_{tag}_z.csv"), stacked_z, delimiter=",", fmt="%.17g")
    np.savetxt(os.path.join(out, f"xc_{tag}_zf.csv"), zf, delimiter=",", fmt="%.17g")
    with open(os.path.join(out, f"xc_{tag}_dims.txt"), "w") as f: f.write(f"{N} {T}")
