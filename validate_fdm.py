import numpy as np


def analytical_gaussian(x, t, x0, sigma0, alpha):
    var = sigma0 * sigma0 + 2.0 * alpha * t
    amp = sigma0 / np.sqrt(var)
    return amp * np.exp(-((x - x0) ** 2) / (2.0 * var))


def solve_pure_diffusion(alpha, L, T_total, x0, sigma0, nx=201):
    dx = L / (nx - 1)
    dt = 0.4 * dx * dx / (2.0 * alpha)
    nt = int(np.ceil(T_total / dt)) + 1
    dt = T_total / (nt - 1)

    x = np.linspace(0.0, L, nx)
    r = alpha * dt / (dx * dx)

    T = np.exp(-((x - x0) ** 2) / (2.0 * sigma0 * sigma0))

    for n in range(0, nt - 1):
        Tnew = T.copy()
        for i in range(1, nx - 1):
            Tnew[i] = T[i] + r * (T[i + 1] - 2.0 * T[i] + T[i - 1])
        Tnew[0] = Tnew[1]
        Tnew[nx - 1] = Tnew[nx - 2]
        T = Tnew

    return x, T


if __name__ == "__main__":
    alpha = 5e-4
    L = 2.0
    T_total = 100.0
    x0 = 1.0
    sigma0 = 0.15

    x, T_num = solve_pure_diffusion(alpha, L, T_total, x0, sigma0)
    T_exact = analytical_gaussian(x, T_total, x0, sigma0, alpha)

    err = np.abs(T_num - T_exact)
    rel = np.linalg.norm(T_num - T_exact) / np.linalg.norm(T_exact)

    print("max abs error", round(float(err.max()), 6))
    print("L2 relative error", round(float(rel), 6))
    print("peak numerical", round(float(T_num.max()), 5))
    print("peak analytical", round(float(T_exact.max()), 5))

    if rel < 0.02:
        print("PASS solver matches analytical diffusion")
    else:
        print("FAIL something is off in the solver")
