import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem
import fdm_2d
from pinn_2d import HardPINN2D

R = fdm_2d.R


def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")

    x_grid, r_grid, times, snaps = fdm_2d.solve_fdm_2d()

    model = HardPINN2D(problem.L, R, problem.T_total, problem.T_ambient)
    model.load_state_dict(torch.load("results/forward_pinn_2d.pth"))
    model.eval()

    nx = len(x_grid)
    nr = len(r_grid)

    T_pinn = np.zeros((nx, nr))
    tfin = float(times[-1])
    for j in range(nr):
        xq = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)
        rq = torch.full((nx, 1), float((r_grid[j] / R) ** 2),
                        dtype=torch.float32)
        tq = torch.full((nx, 1), tfin, dtype=torch.float32)
        with torch.no_grad():
            T_pinn[:, j] = model(xq, rq, tq).numpy().flatten()

    T_fdm = snaps[-1]
    err = np.abs(T_pinn - T_fdm)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    im0 = axes[0].imshow(T_fdm.T, aspect="auto", origin="lower",
                         extent=[0, problem.L, 0, R * 1000], cmap="hot")
    axes[0].set_title("2D FDM at t=50s")
    axes[0].set_ylabel("r (mm)")
    fig.colorbar(im0, ax=axes[0], label="T (C)")

    im1 = axes[1].imshow(T_pinn.T, aspect="auto", origin="lower",
                         extent=[0, problem.L, 0, R * 1000], cmap="hot")
    axes[1].set_title("2D PINN at t=50s")
    axes[1].set_ylabel("r (mm)")
    fig.colorbar(im1, ax=axes[1], label="T (C)")

    im2 = axes[2].imshow(err.T, aspect="auto", origin="lower",
                         extent=[0, problem.L, 0, R * 1000], cmap="viridis")
    axes[2].set_title("abs error (C)")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("r (mm)")
    fig.colorbar(im2, ax=axes[2], label="C")

    plt.tight_layout()
    plt.savefig("figures/pinn2d_vs_fdm2d.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    ic = nx // 2
    plt.plot(r_grid * 1000, T_fdm[ic, :], "o-", label="FDM")
    plt.plot(r_grid * 1000, T_pinn[ic, :], "s--", label="PINN")
    plt.xlabel("r (mm)")
    plt.ylabel("T (C)")
    plt.title("Radial profile at tube centre, t=50s (axis to glass surface)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/radial_profile.png", dpi=120)
    plt.close()

    rel = np.linalg.norm(T_pinn - T_fdm) / np.linalg.norm(T_fdm)
    print("t=50s slice relL2:", round(float(rel), 5))
    print("max abs err:", round(float(err.max()), 4), "C")
    print("saved 2d figures")


if __name__ == "__main__":
    main()
