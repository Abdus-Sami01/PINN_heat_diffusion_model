import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import problem


def main():
    if not os.path.exists("figures"):
        os.makedirs("figures")

    h_hist = np.load("results/joint_h_history.npy")
    P_hist = np.load("results/joint_P_history.npy")
    steps = np.arange(len(h_hist)) * 50

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(steps, h_hist, label="estimated h")
    axes[0].axhline(problem.h_true, color="red", linestyle="--",
                    label="true h = " + str(problem.h_true))
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("h")
    axes[0].set_title("h (started at 0.15)")
    axes[0].legend()

    axes[1].plot(steps, P_hist, color="green", label="estimated P")
    axes[1].axhline(problem.power, color="red", linestyle="--",
                    label="true P = " + str(problem.power))
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("Q peak power")
    axes[1].set_title("source power (started at 4.0)")
    axes[1].legend()

    fig.suptitle("Joint inversion: two unknowns from the same 3 noisy sensors")
    plt.tight_layout()
    plt.savefig("figures/joint_inversion.png", dpi=120)
    plt.close()
    print("saved joint_inversion.png")

    with open("results/joint_inverse.json") as f:
        r = json.load(f)
    print("h err", round(r["h_err_pct"], 2), "% | P err",
          round(r["P_err_pct"], 2), "%")


if __name__ == "__main__":
    main()
