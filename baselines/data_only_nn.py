import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import problem
from pinn_model import PINN

torch.set_num_threads(4)


def main(n_sensors=3, n_times=8, noise_std=1.0, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_grid, t_grid, T_grid = problem.get_ground_truth()
    xs, ts, Ts = problem.sample_sensors(x_grid, t_grid, T_grid,
                                        n_sensors=n_sensors, n_times=n_times,
                                        noise_std=noise_std, seed=seed)

    x_obs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    t_obs = torch.tensor(ts, dtype=torch.float32).view(-1, 1)
    T_obs = torch.tensor(Ts, dtype=torch.float32).view(-1, 1)

    model = PINN(problem.L, problem.T_total, problem.T_ambient,
                 hidden=64, layers=5)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(8000):
        pred = model(x_obs, t_obs)
        loss = torch.mean((pred - T_obs) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 2000 == 0:
            print("epoch", epoch, "data mse", round(float(loss.item()), 5),
                  flush=True)

    nx = len(x_grid)
    nt = len(t_grid)
    T_pred = np.zeros((nx, nt))
    model.eval()
    for j in range(nt):
        xq = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)
        tq = torch.full((nx, 1), float(t_grid[j]), dtype=torch.float32)
        with torch.no_grad():
            T_pred[:, j] = model(xq, tq).numpy().flatten()

    rel = np.linalg.norm(T_pred - T_grid) / np.linalg.norm(T_grid)
    maxerr = np.abs(T_pred - T_grid).max()
    print("data-only NN full grid relL2:", round(float(rel), 5))
    print("data-only NN max abs error:", round(float(maxerr), 3), "C")

    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/baseline_data_only.txt", "w") as f:
        f.write("relative_l2 " + str(rel) + "\n")
        f.write("max_abs_error " + str(float(maxerr)) + "\n")

    np.save("results/baseline_data_only_grid.npy", T_pred)
    return rel


if __name__ == "__main__":
    main()
