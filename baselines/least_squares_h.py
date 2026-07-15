import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import problem
from fdm_reference import solve_fdm, make_Q


def misfit(h_candidate, xs, ts, Ts):
    Q = make_Q(problem.power, problem.center, problem.width)
    x_grid, t_grid, T_grid = solve_fdm(problem.alpha, h_candidate, Q,
                                       problem.T_ambient, problem.L,
                                       problem.T_total)
    total = 0.0
    for k in range(len(xs)):
        val = problem.interp_T(x_grid, t_grid, T_grid, xs[k], ts[k])
        total = total + (val - Ts[k]) ** 2
    return total / len(xs)


def fit_h(xs, ts, Ts, lo=0.005, hi=0.5, iters=40):
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a = lo
    b = hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = misfit(c, xs, ts, Ts)
    fd = misfit(d, xs, ts, Ts)
    for i in range(iters):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = misfit(c, xs, ts, Ts)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = misfit(d, xs, ts, Ts)
    return (a + b) / 2.0


if __name__ == "__main__":
    x_grid, t_grid, T_grid = problem.get_ground_truth()

    print("sensors | noise | h fitted | error %")
    rows = []
    for ns in [5, 3, 1]:
        for nz in [0.0, 1.0, 2.0]:
            xs, ts, Ts = problem.sample_sensors(x_grid, t_grid, T_grid,
                                                n_sensors=ns, n_times=8,
                                                noise_std=nz, seed=1)
            h_fit = fit_h(xs, ts, Ts)
            err = abs(h_fit - problem.h_true) / problem.h_true * 100.0
            print(str(ns) + " | " + str(nz) + " | " +
                  str(round(h_fit, 5)) + " | " + str(round(err, 2)), flush=True)
            rows.append([ns, nz, h_fit, err])

    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/least_squares_h.txt", "w") as f:
        f.write("sensors noise h_fitted err_pct\n")
        for r in rows:
            f.write(str(r[0]) + " " + str(r[1]) + " " +
                    str(round(r[2], 6)) + " " + str(round(r[3], 3)) + "\n")
