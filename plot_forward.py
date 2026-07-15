import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem
from pinn_model import HardPINN


def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")

    model = HardPINN(problem.L, problem.T_total, problem.T_ambient,
                     tau=1.0 / problem.h_true)
    model.load_state_dict(torch.load("results/forward_pinn.pth"))
    model.eval()

    x_grid, t_grid, T_grid = problem.get_ground_truth()

    nx = len(x_grid)
    nt = len(t_grid)

    T_pinn = np.zeros((nx, nt))
    for j in range(nt):
        xt = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)
        tt = torch.full((nx, 1), float(t_grid[j]), dtype=torch.float32)
        with torch.no_grad():
            T_pinn[:, j] = model(xt, tt).numpy().flatten()

    err = np.abs(T_pinn - T_grid)
    rel = np.linalg.norm(T_pinn - T_grid) / np.linalg.norm(T_grid)
    print("full grid relative L2:", round(float(rel), 5))
    print("max abs error:", round(float(err.max()), 4), "C")

    with open("results/forward_fullgrid.txt", "w") as f:
        f.write("relative_l2 " + str(float(rel)) + "\n")
        f.write("max_abs_error " + str(float(err.max())) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    im0 = axes[0].imshow(T_grid, aspect="auto", origin="lower",
                         extent=[t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]],
                         cmap="hot")
    axes[0].set_title("FDM ground truth")
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("x (m)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(T_pinn, aspect="auto", origin="lower",
                         extent=[t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]],
                         cmap="hot")
    axes[1].set_title("Forward PINN")
    axes[1].set_xlabel("t (s)")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(err, aspect="auto", origin="lower",
                         extent=[t_grid[0], t_grid[-1], x_grid[0], x_grid[-1]],
                         cmap="viridis")
    axes[2].set_title("abs error (C)")
    axes[2].set_xlabel("t (s)")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("figures/forward_pinn_vs_fdm.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    for j in [nt // 8, nt // 3, nt - 1]:
        plt.plot(x_grid, T_grid[:, j], "-",
                 label="FDM t=" + str(round(float(t_grid[j]), 1)))
        plt.plot(x_grid, T_pinn[:, j], "--",
                 label="PINN t=" + str(round(float(t_grid[j]), 1)))
    plt.xlabel("x (m)")
    plt.ylabel("T (C)")
    plt.title("PINN vs FDM temperature profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/forward_profiles.png", dpi=120)
    plt.close()

    print("saved forward figures")


if __name__ == "__main__":
    main()
