import numpy as np

import problem

R = 0.01
beta = problem.h_true * R / (2.0 * problem.alpha)


def Q_2d(x):
    return problem.power * np.exp(-((x - problem.center) ** 2) /
                                  (2.0 * problem.width * problem.width))


def solve_fdm_2d(nx=100, nr=21, n_snapshots=251):
    alpha = problem.alpha
    L = problem.L
    T_total = problem.T_total
    Tamb = problem.T_ambient

    dx = L / (nx - 1)
    dr = R / (nr - 1)

    dt_limit = 1.0 / (2.0 * alpha * (1.0 / (dx * dx) + 1.0 / (dr * dr)))
    dt = 0.4 * dt_limit
    nt = int(np.ceil(T_total / dt)) + 1
    dt = T_total / (nt - 1)

    x = np.linspace(0.0, L, nx)
    r = np.linspace(0.0, R, nr)

    T = np.full((nx, nr), Tamb)
    Qx = Q_2d(x).reshape(nx, 1)

    snap_every = max(1, nt // n_snapshots)
    snaps = []
    snap_times = []

    inv_dx2 = 1.0 / (dx * dx)
    inv_dr2 = 1.0 / (dr * dr)

    for n in range(nt):
        if n % snap_every == 0 or n == nt - 1:
            snaps.append(T.copy())
            snap_times.append(n * dt)
        if n == nt - 1:
            break

        Txx = np.zeros((nx, nr))
        Txx[1:-1, :] = (T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]) * inv_dx2
        Txx[0, :] = (T[1, :] - T[0, :]) * 2.0 * inv_dx2
        Txx[-1, :] = (T[-2, :] - T[-1, :]) * 2.0 * inv_dx2

        rad = np.zeros((nx, nr))
        rad[:, 1:-1] = (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]) * inv_dr2
        rmid = r[1:-1].reshape(1, nr - 2)
        rad[:, 1:-1] += (T[:, 2:] - T[:, :-2]) / (2.0 * dr) / rmid

        rad[:, 0] = 4.0 * (T[:, 1] - T[:, 0]) * inv_dr2

        ghost = T[:, -2] - 2.0 * dr * beta * (T[:, -1] - Tamb)
        Trr_surf = (ghost - 2.0 * T[:, -1] + T[:, -2]) * inv_dr2
        Tr_surf = -beta * (T[:, -1] - Tamb)
        rad[:, -1] = Trr_surf + Tr_surf / R

        T = T + dt * (alpha * (Txx + rad) + Qx)

    return x, r, np.array(snap_times), np.array(snaps)


if __name__ == "__main__":
    x, r, times, snaps = solve_fdm_2d()
    print("snapshots", snaps.shape, "dt between", round(times[1] - times[0], 4))
    print("final peak temp", round(float(snaps[-1].max()), 3))
    print("final surface-axis delta at center x:",
          round(float(snaps[-1][50, 0] - snaps[-1][50, -1]), 4), "C")

    T_avg = np.zeros(len(x))
    for i in range(len(x)):
        T_avg[i] = np.trapezoid(snaps[-1][i, :] * r, r) / np.trapezoid(r, r)

    x1, t1, T1 = problem.get_ground_truth()
    rel = np.linalg.norm(T_avg - T1[:, -1]) / np.linalg.norm(T1[:, -1])
    print("radial-avg 2D vs 1D model at t=50s, rel L2:", round(float(rel), 5))
    if rel < 0.02:
        print("CROSS-CHECK PASS 2d radially averages to the 1d model")
    else:
        print("CROSS-CHECK FAIL mismatch between 2d and 1d physics")
