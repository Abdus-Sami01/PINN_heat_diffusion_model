import os
import json
import numpy as np
import torch

import problem
from fdm_reference import solve_fdm, make_Q
from pinn_model import HardPINN

torch.set_num_threads(4)

c_rad_true = 1.2e-10


def get_radiative_ground_truth():
    Q = make_Q(problem.power, problem.center, problem.width)
    return solve_fdm(problem.alpha, problem.h_true, Q, problem.T_ambient,
                     problem.L, problem.T_total, c_rad=c_rad_true)


def pde_residual_rad(model, x, t, h, c_scaled):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    T = model(x, t)
    ones = torch.ones_like(T)

    T_t = torch.autograd.grad(T, t, grad_outputs=ones, create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, grad_outputs=ones, create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                               create_graph=True)[0]

    Q = problem.Q_torch(x, t)
    res = T_t - problem.alpha * T_xx + h * (T - problem.T_ambient) - Q

    if c_scaled is not None:
        TK = T + 273.15
        TaK = problem.T_ambient + 273.15
        res = res + c_scaled * 1e-10 * (TK ** 4 - TaK ** 4)

    return torch.mean(res ** 2)


def inv_softplus(y):
    return float(np.log(np.exp(y) - 1.0))


def train_rad(include_radiation, n_sensors=5, n_times=10, noise_std=0.5,
              adam_epochs=14000, lbfgs_steps=500, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_grid, t_grid, T_grid = get_radiative_ground_truth()
    xs, ts, Ts = problem.sample_sensors(x_grid, t_grid, T_grid,
                                        n_sensors=n_sensors,
                                        n_times=n_times,
                                        noise_std=noise_std, seed=seed)

    x_obs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    t_obs = torch.tensor(ts, dtype=torch.float32).view(-1, 1)
    T_obs = torch.tensor(Ts, dtype=torch.float32).view(-1, 1)

    model = HardPINN(problem.L, problem.T_total, problem.T_ambient, tau=15.0)

    h_raw = torch.nn.Parameter(torch.tensor(inv_softplus(0.15)))
    phys = [h_raw]
    c_raw = None
    if include_radiation:
        c_raw = torch.nn.Parameter(torch.tensor(inv_softplus(3.0)))
        phys.append(c_raw)

    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3},
        {"params": phys, "lr": 2e-2},
    ])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)

    w_data = 10.0
    n_col = 4000

    for epoch in range(adam_epochs):
        h = torch.nn.functional.softplus(h_raw)
        c = torch.nn.functional.softplus(c_raw) if include_radiation else None

        x = torch.rand(n_col, 1) * problem.L
        t = torch.rand(n_col, 1) * problem.T_total

        lp = pde_residual_rad(model, x, t, h, c)
        ld = torch.mean((model(x_obs, t_obs) - T_obs) ** 2)
        loss = lp + w_data * ld

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if verbose and epoch % 2000 == 0:
            msg = "epoch " + str(epoch) + \
                  " pde " + str(round(float(lp.item()), 6)) + \
                  " data " + str(round(float(ld.item()), 6)) + \
                  " h " + str(round(float(h.item()), 5))
            if include_radiation:
                msg += " c " + str(round(float(c.item()), 4)) + "e-10"
            print(msg, flush=True)

    xf = torch.rand(6000, 1) * problem.L
    tf = torch.rand(6000, 1) * problem.T_total
    params = list(model.parameters()) + phys
    lbfgs = torch.optim.LBFGS(params, max_iter=lbfgs_steps,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        h = torch.nn.functional.softplus(h_raw)
        c = torch.nn.functional.softplus(c_raw) if include_radiation else None
        l = pde_residual_rad(model, xf, tf, h, c) + \
            w_data * torch.mean((model(x_obs, t_obs) - T_obs) ** 2)
        l.backward()
        return l

    lbfgs.step(closure)

    h_final = float(torch.nn.functional.softplus(h_raw).item())
    c_final = None
    if include_radiation:
        c_final = float(torch.nn.functional.softplus(c_raw).item()) * 1e-10

    xf2 = torch.rand(4000, 1) * problem.L
    tf2 = torch.rand(4000, 1) * problem.T_total
    h_t = torch.nn.functional.softplus(h_raw)
    c_t = torch.nn.functional.softplus(c_raw) if include_radiation else None
    final_data = float(torch.mean(
        (model(x_obs, t_obs) - T_obs) ** 2).item())

    return h_final, c_final, final_data


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    print("=== STUDY A: radiation-blind model on radiative truth ===",
          flush=True)
    h_blind, _, fit_blind = train_rad(include_radiation=False)
    h_bias = (h_blind - problem.h_true) / problem.h_true * 100.0
    print("blind h:", round(h_blind, 5), "true:", problem.h_true,
          "bias:", round(h_bias, 2), "%  data fit:", round(fit_blind, 5))

    print("", flush=True)
    print("=== STUDY B: radiation-aware model, joint h + c ===", flush=True)
    h_aware, c_aware, fit_aware = train_rad(include_radiation=True)
    h_err = abs(h_aware - problem.h_true) / problem.h_true * 100.0
    c_err = abs(c_aware - c_rad_true) / c_rad_true * 100.0
    print("aware h:", round(h_aware, 5), "err:", round(h_err, 2), "%")
    print("aware c:", "{:.3e}".format(c_aware), "true:",
          "{:.1e}".format(c_rad_true), "err:", round(c_err, 2), "%")

    with open("results/radiation_study.json", "w") as f:
        json.dump({"h_true": problem.h_true, "c_rad_true": c_rad_true,
                   "blind_h": h_blind, "blind_h_bias_pct": h_bias,
                   "blind_data_fit": fit_blind,
                   "aware_h": h_aware, "aware_h_err_pct": h_err,
                   "aware_c": c_aware, "aware_c_err_pct": c_err,
                   "aware_data_fit": fit_aware}, f, indent=2)
    print("RADIATION STUDY DONE")
