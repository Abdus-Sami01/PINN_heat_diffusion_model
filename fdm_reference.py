import numpy as np


def make_Q(power, center, width):
    def Q(x, t):
        return power * np.exp(-((x - center) ** 2) / (2.0 * width * width))
    return Q


def solve_fdm(alpha, h, Q_func, T_ambient, L, T_total, nx=100, nt=None):
    dx = L / (nx - 1)

    dt_stable = dx * dx / (2.0 * alpha)
    dt = 0.4 * dt_stable

    if nt is None:
        nt = int(np.ceil(T_total / dt)) + 1
    else:
        dt = T_total / (nt - 1)

    x = np.linspace(0.0, L, nx)
    t = np.linspace(0.0, T_total, nt)

    T = np.zeros((nx, nt))
    T[:, 0] = T_ambient

    r = alpha * dt / (dx * dx)

    if np.isscalar(h):
        h_arr = np.full(nx, float(h))
    else:
        h_arr = np.asarray(h)

    for n in range(0, nt - 1):
        Tn = T[:, n]
        Tnew = np.zeros(nx)

        for i in range(1, nx - 1):
            diffusion = r * (Tn[i + 1] - 2.0 * Tn[i] + Tn[i - 1])
            convection = h_arr[i] * (Tn[i] - T_ambient) * dt
            source = Q_func(x[i], t[n]) * dt
            Tnew[i] = Tn[i] + diffusion - convection + source

        Tnew[0] = Tnew[1]
        Tnew[nx - 1] = Tnew[nx - 2]

        T[:, n + 1] = Tnew

    return x, t, T


if __name__ == "__main__":
    alpha = 1e-4
    h = 0.05
    T_ambient = 20.0
    L = 1.0
    T_total = 50.0

    Q = make_Q(power=2.0, center=0.5, width=0.15)

    x, t, T = solve_fdm(alpha, h, Q, T_ambient, L, T_total)

    print("grid shape", T.shape)
    print("x points", len(x), "t points", len(t))
    print("max temp", round(float(T.max()), 3))
    print("min temp", round(float(T.min()), 3))
    print("final center temp", round(float(T[len(x) // 2, -1]), 3))
