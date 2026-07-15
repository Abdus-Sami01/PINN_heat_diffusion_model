import os
import json

import train_inverse


def main():
    sensor_counts = [5, 3, 1]
    noise_levels = [0.0, 1.0, 2.0]

    results = []

    for ns in sensor_counts:
        for nz in noise_levels:
            print("=== sensors", ns, "noise", nz, "===", flush=True)
            model, h_final, h_err = train_inverse.train_inverse(
                n_sensors=ns, n_times=8, noise_std=nz,
                adam_epochs=6000, lbfgs_steps=400,
                seed=1, verbose=False)
            print("sensors", ns, "noise", nz,
                  "h", round(h_final, 5), "err%", round(h_err, 2), flush=True)
            results.append({"sensors": ns, "noise": nz,
                            "h_recovered": h_final, "h_err_pct": h_err})

    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    print("")
    print("sensors | noise | h recovered | error %")
    for r in results:
        print(str(r["sensors"]) + " | " + str(r["noise"]) + " | " +
              str(round(r["h_recovered"], 5)) + " | " +
              str(round(r["h_err_pct"], 2)))


if __name__ == "__main__":
    main()
