import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem


def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")

    h_pred = np.load("results/spatial_h_pred.npy")
    h_tru = np.load("results/spatial_h_true.npy")
    x = np.linspace(0, problem.L, len(h_tru))

    sensor_x = np.linspace(0.15, 0.85, 7)

    plt.figure(figsize=(9, 5))
    plt.plot(x, h_tru, "k-", linewidth=2, label="true h(x)")
    plt.plot(x, h_pred, "r--", linewidth=2, label="recovered h(x)")
    for i, sx in enumerate(sensor_x):
        plt.axvline(sx, color="gray", alpha=0.3,
                    label="sensor positions" if i == 0 else None)
    plt.xlabel("x (m)")
    plt.ylabel("h(x)")
    plt.title("Recovering a convection PROFILE from 7 noisy thermocouples")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/spatial_h_recovery.png", dpi=120)
    plt.close()

    rel = np.linalg.norm(h_pred - h_tru) / np.linalg.norm(h_tru)
    print("h(x) relL2:", round(float(rel), 5))
    print("saved spatial_h_recovery.png")


if __name__ == "__main__":
    main()
