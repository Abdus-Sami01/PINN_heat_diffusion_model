import os
import json
import numpy as np

import train_inverse


def main():
    sensor_counts = [5, 3, 1]
    noise_levels = [0.0, 1.0, 2.0]
    seeds = [1, 2, 3]

    cells = []

    for ns in sensor_counts:
        for nz in noise_levels:
            errs = []
            hs = []
            for s in seeds:
                print("=== sensors", ns, "noise", nz, "seed", s, "===",
                      flush=True)
                model, h_final, h_err, hist = train_inverse.train_inverse(
                    n_sensors=ns, n_times=8, noise_std=nz,
                    adam_epochs=6000, lbfgs_steps=400,
                    seed=s, verbose=False)
                print("  h", round(h_final, 5), "err%", round(h_err, 2),
                      flush=True)
                errs.append(h_err)
                hs.append(h_final)

            cell = {"sensors": ns, "noise": nz,
                    "h_values": hs,
                    "err_values": errs,
                    "err_mean": float(np.mean(errs)),
                    "err_std": float(np.std(errs))}
            cells.append(cell)
            print("cell done: sensors", ns, "noise", nz,
                  "err mean", round(cell["err_mean"], 2),
                  "+/-", round(cell["err_std"], 2), flush=True)

    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/multiseed_sensitivity.json", "w") as f:
        json.dump(cells, f, indent=2)

    print("")
    print("sensors | noise | h err mean % | h err std %")
    for c in cells:
        print(str(c["sensors"]) + " | " + str(c["noise"]) + " | " +
              str(round(c["err_mean"], 2)) + " | " +
              str(round(c["err_std"], 2)))
    print("MULTISEED DONE")


if __name__ == "__main__":
    main()
