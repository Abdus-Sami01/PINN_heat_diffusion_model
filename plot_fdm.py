import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fdm_reference import solve_fdm, make_Q


def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")

    alpha = 1e-4
    h = 0.05
    T_ambient = 20.0
    L = 1.0
    T_total = 50.0

    Q = make_Q(power=2.0, center=0.5, width=0.15)
    x, t, T = solve_fdm(alpha, h, Q, T_ambient, L, T_total)

    plt.figure(figsize=(8, 5))
    plt.imshow(T, aspect="auto", origin="lower",
               extent=[t[0], t[-1], x[0], x[-1]], cmap="hot")
    plt.colorbar(label="Temperature (C)")
    plt.xlabel("time (s)")
    plt.ylabel("position x (m)")
    plt.title("FDM heat diffusion along tube")
    plt.tight_layout()
    plt.savefig("figures/fdm_heatmap.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    time_idx = [0, len(t) // 8, len(t) // 4, len(t) // 2, len(t) - 1]
    for idx in time_idx:
        plt.plot(x, T[:, idx], label="t=" + str(round(float(t[idx]), 1)) + "s")
    plt.xlabel("position x (m)")
    plt.ylabel("Temperature (C)")
    plt.title("Temperature profiles at different times")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/fdm_time_slices.png", dpi=120)
    plt.close()

    center = len(x) // 2
    plt.figure(figsize=(8, 5))
    plt.plot(t, T[center, :])
    plt.xlabel("time (s)")
    plt.ylabel("Temperature (C)")
    plt.title("Warm-up curve at tube center")
    plt.tight_layout()
    plt.savefig("figures/fdm_warmup.png", dpi=120)
    plt.close()

    print("saved figures to figures/")


if __name__ == "__main__":
    main()
