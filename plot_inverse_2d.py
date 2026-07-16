import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fdm_2d


def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")

    hist = np.load("results/beta_history.npy")
    steps = np.arange(len(hist)) * 50

    plt.figure(figsize=(8, 5))
    plt.plot(steps, hist, label="estimated beta")
    plt.axhline(fdm_2d.beta, color="red", linestyle="--",
                label="true beta = " + str(round(fdm_2d.beta, 2)))
    plt.xlabel("training step")
    plt.ylabel("beta (1/m)")
    plt.title("2D inverse: surface coefficient from surface-only sensors")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/beta_convergence.png", dpi=120)
    plt.close()

    with open("results/inverse_2d.json") as f:
        r = json.load(f)
    print("beta err", round(r["beta_err_pct"], 2), "%")
    print("saved beta_convergence.png")


if __name__ == "__main__":
    main()
