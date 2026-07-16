import os
import json
import numpy as np
import torch

import problem
import fdm_2d
from pinn_2d import HardPINN2D
from train_forward_2d import Q_torch_x

torch.set_num_threads(4)

R = fdm_2d.R
beta_true = fdm_2d.beta


def pde_loss_2d_free(model, x, rho, t):
    x = x.clone().detach().requires_grad_(True)
    rho = rho.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    T = model(x, rho, t)
    ones = torch.ones_like(T)

    T_t = torch.autograd.grad(T, t, grad_outputs=ones, create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, grad_outputs=ones, create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                               create_graph=True)[0]
    T_rho = torch.autograd.grad(T, rho, grad_outputs=ones,
                                create_graph=True)[0]
    T_rhorho = torch.autograd.grad(T_rho, rho,
                                   grad_outputs=torch.ones_like(T_rho),
                                   create_graph=True)[0]

    radial = (4.0 / (R * R)) * (rho * T_rhorho + T_rho)
    res = T_t - problem.alpha * (T_xx + radial) - Q_torch_x(x)
    return torch.mean(res ** 2)


def robin_loss_free(model, x, t, beta):
    rho = torch.ones_like(x).requires_grad_(True)
    T = model(x, rho, t)
    T_rho = torch.autograd.grad(T, rho, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]
    target = -(beta * R / 2.0) * (T - problem.T_ambient)
    return torch.mean((T_rho - target) ** 2)


def sample_surface_sensors(x_grid, r_grid, times, snaps,
                           n_sensors=3, n_times=8, noise_std=1.0, seed=0):
    rng = np.random.default_rng(seed)
    sensor_x = np.linspace(0.15, 0.85, n_sensors)
    sensor_t = np.linspace(2.0, problem.T_total, n_times)

    xs = []
    ts = []
    Ts = []
    for sx in sensor_x:
        xi = int(np.argmin(np.abs(x_grid - sx)))
        for st in sensor_t:
            si = int(np.argmin(np.abs(times - st)))
            val = snaps[si][xi, -1]
            if noise_std > 0:
                val = val + rng.normal(0.0, noise_std)
            xs.append(x_grid[xi])
            ts.append(times[si])
            Ts.append(val)
    return np.array(xs), np.array(ts), np.array(Ts)


def inv_softplus(y):
    return float(np.log(np.exp(y) - 1.0))


def train_inverse_2d(n_sensors=3, n_times=8, noise_std=1.0, beta_init=7.5,
                     adam_epochs=16000, lbfgs_steps=500, seed=0,
                     verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("computing 2d fdm ground truth...", flush=True)
    x_grid, r_grid, times, snaps = fdm_2d.solve_fdm_2d()
    print("done, true beta =", round(beta_true, 4), flush=True)

    xs, ts, Ts = sample_surface_sensors(x_grid, r_grid, times, snaps,
                                        n_sensors, n_times, noise_std, seed)

    x_obs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    rho_obs = torch.ones_like(x_obs)
    t_obs = torch.tensor(ts, dtype=torch.float32).view(-1, 1)
    T_obs = torch.tensor(Ts, dtype=torch.float32).view(-1, 1)

    model = HardPINN2D(problem.L, R, problem.T_total, problem.T_ambient)

    b_raw = torch.nn.Parameter(torch.tensor(inv_softplus(beta_init)))

    params = list(model.parameters()) + [b_raw]
    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3},
        {"params": [b_raw], "lr": 2e-2},
    ])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)

    w_bc = 10.0
    w_data = 10.0
    n_col = 4000
    n_bc = 500

    b_hist = []

    for epoch in range(adam_epochs):
        beta = torch.nn.functional.softplus(b_raw)

        x = torch.rand(n_col, 1) * problem.L
        rho = torch.rand(n_col, 1)
        t = torch.rand(n_col, 1) * problem.T_total
        xb = torch.rand(n_bc, 1) * problem.L
        tb = torch.rand(n_bc, 1) * problem.T_total

        lp = pde_loss_2d_free(model, x, rho, t)
        lb = robin_loss_free(model, xb, tb, beta)
        ld = torch.mean((model(x_obs, rho_obs, t_obs) - T_obs) ** 2)

        loss = lp + w_bc * lb + w_data * ld

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 50 == 0:
            b_hist.append(float(beta.item()))

        if verbose and epoch % 1000 == 0:
            print("epoch", epoch, "pde", round(float(lp.item()), 6),
                  "robin", round(float(lb.item()), 8),
                  "data", round(float(ld.item()), 6),
                  "beta", round(float(beta.item()), 4), flush=True)

    torch.save(model.state_dict(), "results/inverse_pinn_2d_adam.pth")

    xf = torch.rand(6000, 1) * problem.L
    rhof = torch.rand(6000, 1)
    tf = torch.rand(6000, 1) * problem.T_total
    xbf = torch.rand(800, 1) * problem.L
    tbf = torch.rand(800, 1) * problem.T_total

    lbfgs = torch.optim.LBFGS(params, max_iter=lbfgs_steps,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        beta = torch.nn.functional.softplus(b_raw)
        l = pde_loss_2d_free(model, xf, rhof, tf) + \
            w_bc * robin_loss_free(model, xbf, tbf, beta) + \
            w_data * torch.mean((model(x_obs, rho_obs, t_obs) - T_obs) ** 2)
        l.backward()
        return l

    lbfgs.step(closure)

    b_final = float(torch.nn.functional.softplus(b_raw).item())
    b_hist.append(b_final)
    b_err = abs(b_final - beta_true) / beta_true * 100.0

    h_equiv = 2.0 * problem.alpha * b_final / R
    h_err = abs(h_equiv - problem.h_true) / problem.h_true * 100.0

    print("recovered beta:", round(b_final, 4), "true:", round(beta_true, 4),
          "err:", round(b_err, 2), "%")
    print("equivalent 1d h:", round(h_equiv, 5), "true:", problem.h_true,
          "err:", round(h_err, 2), "%")

    return b_final, b_err, h_equiv, b_hist


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    b_final, b_err, h_equiv, b_hist = train_inverse_2d()

    with open("results/inverse_2d.json", "w") as f:
        json.dump({"beta_recovered": b_final, "beta_true": beta_true,
                   "beta_err_pct": b_err,
                   "h_equivalent": h_equiv,
                   "h_true": problem.h_true}, f, indent=2)
    np.save("results/beta_history.npy", np.array(b_hist))
