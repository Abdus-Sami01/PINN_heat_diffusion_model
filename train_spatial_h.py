import os
import json
import numpy as np
import torch
import torch.nn as nn

import problem
from fdm_reference import solve_fdm, make_Q
from pinn_model import HardPINN

torch.set_num_threads(4)


def h_true_profile(x):
    ramp = 0.04 + 0.02 * (x / problem.L)
    dip = 0.02 * np.exp(-((x - 0.3) ** 2) / (2.0 * 0.1 * 0.1))
    return ramp - dip


class HNet(nn.Module):
    def __init__(self, L, hidden=32):
        super(HNet, self).__init__()
        self.L = L
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        xn = x / self.L * 2.0 - 1.0
        return 0.15 * torch.sigmoid(self.net(xn))


def get_spatial_ground_truth():
    Q = make_Q(problem.power, problem.center, problem.width)
    x_probe = np.linspace(0.0, problem.L, 100)
    h_arr = h_true_profile(x_probe)
    x, t, T = solve_fdm(problem.alpha, h_arr, Q, problem.T_ambient,
                        problem.L, problem.T_total)
    return x, t, T


def pde_loss_spatial(model, h_net, x, t):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    T = model(x, t)
    ones = torch.ones_like(T)

    T_t = torch.autograd.grad(T, t, grad_outputs=ones, create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, grad_outputs=ones, create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                               create_graph=True)[0]

    h = h_net(x)
    Q = problem.Q_torch(x, t)
    res = T_t - problem.alpha * T_xx + h * (T - problem.T_ambient) - Q
    return torch.mean(res ** 2)


def smoothness_loss(h_net, x):
    x = x.clone().detach().requires_grad_(True)
    h = h_net(x)
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h),
                              create_graph=True)[0]
    return torch.mean(h_x ** 2)


def train_spatial(n_sensors=7, n_times=10, noise_std=0.5,
                  adam_epochs=16000, lbfgs_steps=500, seed=0,
                  w_reg=5.0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_grid, t_grid, T_grid = get_spatial_ground_truth()
    xs, ts, Ts = problem.sample_sensors(x_grid, t_grid, T_grid,
                                        n_sensors=n_sensors,
                                        n_times=n_times,
                                        noise_std=noise_std, seed=seed)

    x_obs = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    t_obs = torch.tensor(ts, dtype=torch.float32).view(-1, 1)
    T_obs = torch.tensor(Ts, dtype=torch.float32).view(-1, 1)

    model = HardPINN(problem.L, problem.T_total, problem.T_ambient, tau=15.0)
    h_net = HNet(problem.L)

    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1e-3},
        {"params": h_net.parameters(), "lr": 5e-3},
    ])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)

    w_data = 10.0
    n_col = 4000

    for epoch in range(adam_epochs):
        x = torch.rand(n_col, 1) * problem.L
        t = torch.rand(n_col, 1) * problem.T_total

        lp = pde_loss_spatial(model, h_net, x, t)
        ld = torch.mean((model(x_obs, t_obs) - T_obs) ** 2)
        lr_ = smoothness_loss(h_net, x)
        loss = lp + w_data * ld + w_reg * lr_

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if verbose and epoch % 2000 == 0:
            with torch.no_grad():
                xq = torch.linspace(0, problem.L, 50).view(-1, 1)
                h_pred = h_net(xq).numpy().flatten()
            h_tru = h_true_profile(np.linspace(0, problem.L, 50))
            herr = np.linalg.norm(h_pred - h_tru) / np.linalg.norm(h_tru)
            print("epoch", epoch, "pde", round(float(lp.item()), 6),
                  "data", round(float(ld.item()), 6),
                  "h(x) relL2", round(float(herr), 4), flush=True)

    torch.save({"model": model.state_dict(), "h_net": h_net.state_dict()},
               "results/spatial_h_adam.pth")

    xf = torch.rand(6000, 1) * problem.L
    tf = torch.rand(6000, 1) * problem.T_total
    params = list(model.parameters()) + list(h_net.parameters())
    lbfgs = torch.optim.LBFGS(params, max_iter=lbfgs_steps,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = pde_loss_spatial(model, h_net, xf, tf) + \
            w_data * torch.mean((model(x_obs, t_obs) - T_obs) ** 2) + \
            w_reg * smoothness_loss(h_net, xf)
        l.backward()
        return l

    lbfgs.step(closure)

    xq = torch.linspace(0, problem.L, 100).view(-1, 1)
    with torch.no_grad():
        h_pred = h_net(xq).numpy().flatten()
    xnp = np.linspace(0, problem.L, 100)
    h_tru = h_true_profile(xnp)
    herr = np.linalg.norm(h_pred - h_tru) / np.linalg.norm(h_tru)

    span = (xnp >= 0.15) & (xnp <= 0.85)
    herr_span = np.linalg.norm(h_pred[span] - h_tru[span]) / \
        np.linalg.norm(h_tru[span])

    print("h(x) recovery relative L2 full domain:", round(float(herr), 5))
    print("h(x) recovery relative L2 sensor span:", round(float(herr_span), 5))

    if not os.path.exists("results"):
        os.makedirs("results")
    torch.save({"model": model.state_dict(), "h_net": h_net.state_dict()},
               "results/spatial_h.pth")
    np.save("results/spatial_h_pred.npy", h_pred)
    np.save("results/spatial_h_true.npy", h_tru)
    with open("results/spatial_h_metrics.txt", "w") as f:
        f.write("h_profile_rel_l2 " + str(float(herr)) + "\n")
        f.write("h_profile_rel_l2_sensor_span " + str(float(herr_span)) + "\n")

    return herr, herr_span


if __name__ == "__main__":
    train_spatial()
