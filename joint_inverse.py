import os
import json
import numpy as np
import torch

import problem
from pinn_model import HardPINN
from losses import data_loss

torch.set_num_threads(4)


def pde_loss_joint(model, x, t, alpha, h, P, T_ambient):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    T = model(x, t)

    T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T),
                              create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                              create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                               create_graph=True)[0]

    Q = P * torch.exp(-((x - problem.center) ** 2) /
                      (2.0 * problem.width * problem.width))
    res = T_t - alpha * T_xx + h * (T - T_ambient) - Q
    return torch.mean(res ** 2)


def inv_softplus(y):
    return float(np.log(np.exp(y) - 1.0))


def train_joint(n_sensors=3, n_times=8, noise_std=1.0,
                h_init=0.15, P_init=4.0,
                adam_epochs=12000, lbfgs_steps=800, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_grid, t_grid, T_grid = problem.get_ground_truth()
    xs, ts, Ts = problem.sample_sensors(x_grid, t_grid, T_grid,
                                        n_sensors=n_sensors,
                                        n_times=n_times,
                                        noise_std=noise_std,
                                        seed=seed)

    x_obs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    t_obs = torch.tensor(ts, dtype=torch.float32).view(-1, 1)
    T_obs = torch.tensor(Ts, dtype=torch.float32).view(-1, 1)

    model = HardPINN(problem.L, problem.T_total, problem.T_ambient, tau=15.0)

    h_raw = torch.nn.Parameter(torch.tensor(inv_softplus(h_init)))
    P_raw = torch.nn.Parameter(torch.tensor(inv_softplus(P_init)))

    params = list(model.parameters()) + [h_raw, P_raw]
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.5)

    w_data = 10.0
    n_col = 4000

    h_hist = []
    P_hist = []

    for epoch in range(adam_epochs):
        h = torch.nn.functional.softplus(h_raw)
        P = torch.nn.functional.softplus(P_raw)

        x = torch.rand(n_col, 1) * problem.L
        t = torch.rand(n_col, 1) * problem.T_total

        lp = pde_loss_joint(model, x, t, problem.alpha, h, P,
                            problem.T_ambient)
        ld = data_loss(model, x_obs, t_obs, T_obs)

        loss = lp + w_data * ld

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 50 == 0:
            h_hist.append(float(h.item()))
            P_hist.append(float(P.item()))

        if verbose and epoch % 1000 == 0:
            print("epoch", epoch, "pde", round(float(lp.item()), 6),
                  "data", round(float(ld.item()), 6),
                  "h", round(float(h.item()), 5),
                  "P", round(float(P.item()), 4), flush=True)

    xf = torch.rand(6000, 1) * problem.L
    tf = torch.rand(6000, 1) * problem.T_total

    lbfgs = torch.optim.LBFGS(params, max_iter=lbfgs_steps,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        h = torch.nn.functional.softplus(h_raw)
        P = torch.nn.functional.softplus(P_raw)
        lp = pde_loss_joint(model, xf, tf, problem.alpha, h, P,
                            problem.T_ambient)
        ld = data_loss(model, x_obs, t_obs, T_obs)
        l = lp + w_data * ld
        l.backward()
        return l

    lbfgs.step(closure)

    h_final = float(torch.nn.functional.softplus(h_raw).item())
    P_final = float(torch.nn.functional.softplus(P_raw).item())
    h_hist.append(h_final)
    P_hist.append(P_final)

    h_err = abs(h_final - problem.h_true) / problem.h_true * 100.0
    P_err = abs(P_final - problem.power) / problem.power * 100.0

    print("recovered h:", round(h_final, 5), "true:", problem.h_true,
          "err:", round(h_err, 2), "%")
    print("recovered P:", round(P_final, 4), "true:", problem.power,
          "err:", round(P_err, 2), "%")

    return h_final, P_final, h_err, P_err, h_hist, P_hist


if __name__ == "__main__":
    h_final, P_final, h_err, P_err, h_hist, P_hist = train_joint()

    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/joint_inverse.json", "w") as f:
        json.dump({"h_recovered": h_final, "h_true": problem.h_true,
                   "h_err_pct": h_err,
                   "P_recovered": P_final, "P_true": problem.power,
                   "P_err_pct": P_err}, f, indent=2)
    np.save("results/joint_h_history.npy", np.array(h_hist))
    np.save("results/joint_P_history.npy", np.array(P_hist))
