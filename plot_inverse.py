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

    if os.path.exists("results/h_history.npy"):
        hist = np.load("results/h_history.npy")
        steps = np.arange(len(hist)) * 50
        plt.figure(figsize=(8, 5))
        plt.plot(steps, hist, label="estimated h")
        plt.axhline(problem.h_true, color="red", linestyle="--",
                    label="true h = " + str(problem.h_true))
        plt.xlabel("training step")
        plt.ylabel("h")
        plt.title("Inverse PINN convergence of h (started at 0.15)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/h_convergence.png", dpi=120)
        plt.close()
        print("saved h_convergence.png")

    if os.path.exists("results/sensitivity.json"):
        with open("results/sensitivity.json") as f:
            sens = json.load(f)

        ls_rows = []
        if os.path.exists("results/least_squares_h.txt"):
            with open("results/least_squares_h.txt") as f:
                lines = f.readlines()[1:]
            for line in lines:
                parts = line.split()
                ls_rows.append([int(parts[0]), float(parts[1]),
                                float(parts[2]), float(parts[3])])

        noise_levels = [0.0, 1.0, 2.0]
        sensor_counts = [5, 3, 1]

        plt.figure(figsize=(9, 5))
        width = 0.25
        xpos = np.arange(len(sensor_counts))
        for k in range(len(noise_levels)):
            nz = noise_levels[k]
            errs = []
            for ns in sensor_counts:
                for r in sens:
                    if r["sensors"] == ns and r["noise"] == nz:
                        errs.append(r["h_err_pct"])
            plt.bar(xpos + (k - 1) * width, errs, width,
                    label="noise " + str(nz) + " C")
        plt.xticks(xpos, [str(s) + " sensors" for s in sensor_counts])
        plt.ylabel("h recovery error (%)")
        plt.title("Inverse PINN: h recovery vs sensor count and noise")
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/sensitivity_h_recovery.png", dpi=120)
        plt.close()
        print("saved sensitivity_h_recovery.png")

        if len(ls_rows) > 0:
            plt.figure(figsize=(9, 5))
            pinn_errs = []
            ls_errs = []
            labels = []
            for ns in sensor_counts:
                for nz in noise_levels:
                    for r in sens:
                        if r["sensors"] == ns and r["noise"] == nz:
                            pinn_errs.append(r["h_err_pct"])
                    for row in ls_rows:
                        if row[0] == ns and row[1] == nz:
                            ls_errs.append(row[3])
                    labels.append(str(ns) + "s/" + str(nz) + "C")
            xp = np.arange(len(labels))
            plt.bar(xp - 0.2, pinn_errs, 0.4, label="inverse PINN")
            plt.bar(xp + 0.2, ls_errs, 0.4, label="FDM least squares")
            plt.xticks(xp, labels, rotation=45)
            plt.ylabel("h recovery error (%)")
            plt.title("PINN vs classical least-squares h estimation")
            plt.legend()
            plt.tight_layout()
            plt.savefig("figures/pinn_vs_leastsquares.png", dpi=120)
            plt.close()
            print("saved pinn_vs_leastsquares.png")


if __name__ == "__main__":
    main()
