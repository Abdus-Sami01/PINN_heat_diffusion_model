import os
import json

import problem
import train_radiation


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    print("=== STUDY C: hot tube (2x power), joint h + c ===", flush=True)
    h_hot, c_hot, fit_hot = train_radiation.train_rad(
        include_radiation=True, power=4.0, noise_std=0.25)

    h_err = abs(h_hot - problem.h_true) / problem.h_true * 100.0
    c_err = abs(c_hot - train_radiation.c_rad_true) / \
        train_radiation.c_rad_true * 100.0

    print("hot h:", round(h_hot, 5), "err:", round(h_err, 2), "%")
    print("hot c:", "{:.3e}".format(c_hot), "err:", round(c_err, 2), "%")

    with open("results/radiation_hot.json", "w") as f:
        json.dump({"h": h_hot, "h_err_pct": h_err,
                   "c": c_hot, "c_err_pct": c_err,
                   "data_fit": fit_hot}, f, indent=2)
    print("HOT STUDY DONE")
