import numpy as np
import torch

from fdm_reference import solve_fdm, make_Q


alpha = 1e-4
h_true = 0.05
T_ambient = 20.0
L = 1.0
T_total = 50.0

power = 2.0
center = 0.5
width = 0.15


def Q_numpy(x, t):
    return power * np.exp(-((x - center) ** 2) / (2.0 * width * width))


def Q_torch(x, t):
    return power * torch.exp(-((x - center) ** 2) / (2.0 * width * width))


def get_ground_truth():
    Q = make_Q(power, center, width)
    x, t, T = solve_fdm(alpha, h_true, Q, T_ambient, L, T_total)
    return x, t, T


def interp_T(x_grid, t_grid, T_grid, xq, tq):
    xi = np.clip(np.searchsorted(x_grid, xq) - 1, 0, len(x_grid) - 2)
    ti = np.clip(np.searchsorted(t_grid, tq) - 1, 0, len(t_grid) - 2)

    x0 = x_grid[xi]
    x1 = x_grid[xi + 1]
    t0 = t_grid[ti]
    t1 = t_grid[ti + 1]

    wx = (xq - x0) / (x1 - x0)
    wt = (tq - t0) / (t1 - t0)

    T00 = T_grid[xi, ti]
    T10 = T_grid[xi + 1, ti]
    T01 = T_grid[xi, ti + 1]
    T11 = T_grid[xi + 1, ti + 1]

    top = T00 * (1 - wx) + T10 * wx
    bot = T01 * (1 - wx) + T11 * wx
    return top * (1 - wt) + bot * wt


def sample_sensors(x_grid, t_grid, T_grid, n_sensors=3, n_times=8,
                   noise_std=0.0, seed=0):
    rng = np.random.default_rng(seed)

    sensor_x = np.linspace(0.15, 0.85, n_sensors)
    sensor_t = np.linspace(2.0, T_total, n_times)

    xs = []
    ts = []
    Ts = []
    for sx in sensor_x:
        for st in sensor_t:
            val = interp_T(x_grid, t_grid, T_grid, sx, st)
            if noise_std > 0:
                val = val + rng.normal(0.0, noise_std)
            xs.append(sx)
            ts.append(st)
            Ts.append(val)

    return np.array(xs), np.array(ts), np.array(Ts)
