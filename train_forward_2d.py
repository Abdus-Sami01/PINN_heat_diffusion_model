import os
import numpy as np
import torch

import problem
import fdm_2d
from pinn_2d import HardPINN2D

torch.set_num_threads(4)

R = fdm_2d.R
beta = fdm_2d.beta


def Q_torch_x(x):
    return problem.power * torch.exp(-((x - problem.center) ** 2) /
                                     (2.0 * problem.width * problem.width))


def pde_loss_2d(model, x, rho, t):
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


def robin_loss(model, x, t):
    rho = torch.ones_like(x).requires_grad_(True)
    T = model(x, rho, t)
    T_rho = torch.autograd.grad(T, rho, grad_outputs=torch.ones_like(T),
                                create_graph=True)[0]
    target = -(beta * R / 2.0) * (T - problem.T_ambient)
    return torch.mean((T_rho - target) ** 2)


def relative_l2(model, x_grid, r_grid, times, snaps):
    xs = []
    rhos = []
    ts = []
    truth = []
    for si in range(0, len(times), 25):
        for i in range(0, len(x_grid), 5):
            for j in range(0, len(r_grid), 4):
                xs.append(x_grid[i])
                rhos.append((r_grid[j] / R) ** 2)
                ts.append(times[si])
                truth.append(snaps[si][i, j])

    xt = torch.tensor(np.array(xs), dtype=torch.float32).view(-1, 1)
    rt = torch.tensor(np.array(rhos), dtype=torch.float32).view(-1, 1)
    tt = torch.tensor(np.array(ts), dtype=torch.float32).view(-1, 1)
    truth = np.array(truth)

    model.eval()
    with torch.no_grad():
        pred = model(xt, rt, tt).numpy().flatten()
    model.train()

    return np.linalg.norm(pred - truth) / np.linalg.norm(truth)


def train(adam_epochs=12000, lbfgs_steps=800, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("computing 2d fdm ground truth...", flush=True)
    x_grid, r_grid, times, snaps = fdm_2d.solve_fdm_2d()
    print("done", flush=True)

    model = HardPINN2D(problem.L, R, problem.T_total, problem.T_ambient)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.5)

    n_col = 4000
    n_bc = 500
    w_bc = 10.0

    for epoch in range(adam_epochs):
        x = torch.rand(n_col, 1) * problem.L
        rho = torch.rand(n_col, 1)
        t = torch.rand(n_col, 1) * problem.T_total

        xb = torch.rand(n_bc, 1) * problem.L
        tb = torch.rand(n_bc, 1) * problem.T_total

        lp = pde_loss_2d(model, x, rho, t)
        lb = robin_loss(model, xb, tb)

        loss = lp + w_bc * lb

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if verbose and epoch % 1000 == 0:
            rel = relative_l2(model, x_grid, r_grid, times, snaps)
            print("epoch", epoch, "pde", round(float(lp.item()), 6),
                  "robin", round(float(lb.item()), 8),
                  "relL2", round(rel, 5), flush=True)

    xf = torch.rand(6000, 1) * problem.L
    rhof = torch.rand(6000, 1)
    tf = torch.rand(6000, 1) * problem.T_total
    xbf = torch.rand(800, 1) * problem.L
    tbf = torch.rand(800, 1) * problem.T_total

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = pde_loss_2d(model, xf, rhof, tf) + \
            w_bc * robin_loss(model, xbf, tbf)
        l.backward()
        return l

    lbfgs.step(closure)

    rel = relative_l2(model, x_grid, r_grid, times, snaps)
    print("final relative L2 error vs 2D FDM:", round(rel, 5))

    if rel < 0.03:
        print("GATE PASS 2d forward pinn under 3 percent")
    else:
        print("GATE FAIL 2d still above 3 percent")

    if not os.path.exists("results"):
        os.makedirs("results")
    torch.save(model.state_dict(), "results/forward_pinn_2d.pth")
    with open("results/forward_2d_metrics.txt", "w") as f:
        f.write("relative_l2 " + str(rel) + "\n")

    return model, rel


if __name__ == "__main__":
    train()
