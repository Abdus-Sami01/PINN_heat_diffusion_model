import os
import json
import numpy as np

import problem
import train_inverse


def main():
    seeds = [0, 1, 2, 3, 4]

    h_values = []
    for s in seeds:
        print("=== ensemble member seed", s, "===", flush=True)
        model, h_final, h_err, hist = train_inverse.train_inverse(
            n_sensors=3, n_times=8, noise_std=1.0,
            adam_epochs=6000, lbfgs_steps=400,
            seed=s, verbose=False)
        print("seed", s, "h", round(h_final, 5),
              "err%", round(h_err, 2), flush=True)
        h_values.append(h_final)

    h_arr = np.array(h_values)
    mean = float(h_arr.mean())
    std = float(h_arr.std())

    print("")
    print("ensemble h estimates:", [round(v, 5) for v in h_values])
    print("mean h:", round(mean, 5))
    print("std h:", round(std, 5))
    print("true h:", problem.h_true)
    print("mean error:", round(abs(mean - problem.h_true) / problem.h_true * 100, 2),
          "percent")

    covered = abs(mean - problem.h_true) <= 2 * std
    print("true h within mean +/- 2 std:", covered)

    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/ensemble_uq.json", "w") as f:
        json.dump({"seeds": seeds, "h_values": h_values,
                   "mean": mean, "std": std,
                   "h_true": problem.h_true,
                   "covered_2std": bool(covered)}, f, indent=2)


if __name__ == "__main__":
    main()
