import os
import numpy as np
import torch

import problem
from pinn_model import PINN
from losses import pde_loss, bc_loss, ic_loss, grad_norm


def sample_collocation(n, device):
    x = torch.rand(n, 1) * problem.L
    t = torch.rand(n, 1) * problem.T_total
    return x.to(device), t.to(device)


def sample_boundary(n, device):
    t = torch.rand(n, 1) * problem.T_total
    return t.to(device)


def sample_initial(n, device):
    x = torch.rand(n, 1) * problem.L
    return x.to(device)


def relative_l2(model, device):
    x_grid, t_grid, T_grid = problem.get_ground_truth()

    xs = []
    ts = []
    truth = []
    for i in range(0, len(x_grid), 5):
        for j in range(0, len(t_grid), 10):
            xs.append(x_grid[i])
            ts.append(t_grid[j])
            truth.append(T_grid[i, j])

    xt = torch.tensor(np.array(xs), dtype=torch.float32).view(-1, 1).to(device)
    tt = torch.tensor(np.array(ts), dtype=torch.float32).view(-1, 1).to(device)
    truth = np.array(truth)

    model.eval()
    with torch.no_grad():
        pred = model(xt, tt).cpu().numpy().flatten()
    model.train()

    return np.linalg.norm(pred - truth) / np.linalg.norm(truth)


def train(adam_epochs=8000, lbfgs_steps=300, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PINN(problem.L, problem.T_total, problem.T_ambient).to(device)

    w_pde = 1.0
    w_bc = 10.0
    w_ic = 10.0

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_col = 3000
    n_bc = 200
    n_ic = 200

    for epoch in range(adam_epochs):
        xc, tc = sample_collocation(n_col, device)
        t_bc = sample_boundary(n_bc, device)
        x_ic = sample_initial(n_ic, device)

        lp = pde_loss(model, xc, tc, problem.alpha, problem.h_true,
                      problem.Q_torch, problem.T_ambient)
        lb = bc_loss(model, t_bc, problem.L)
        li = ic_loss(model, x_ic, problem.T_ambient)

        if epoch % 500 == 0 and epoch > 0:
            gp = grad_norm(lp, model)
            gb = grad_norm(lb, model)
            gi = grad_norm(li, model)
            total = gp + gb + gi
            w_pde = float((total / (gp + 1e-8)).item())
            w_bc = float((total / (gb + 1e-8)).item())
            w_ic = float((total / (gi + 1e-8)).item())

        loss = w_pde * lp + w_bc * lb + w_ic * li

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and epoch % 500 == 0:
            rel = relative_l2(model, device)
            print("epoch", epoch, "loss", round(float(loss.item()), 5),
                  "pde", round(float(lp.item()), 5),
                  "bc", round(float(lb.item()), 5),
                  "ic", round(float(li.item()), 5),
                  "relL2", round(rel, 4))

    xc, tc = sample_collocation(n_col, device)
    t_bc = sample_boundary(n_bc, device)
    x_ic = sample_initial(n_ic, device)

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        lp = pde_loss(model, xc, tc, problem.alpha, problem.h_true,
                      problem.Q_torch, problem.T_ambient)
        lb = bc_loss(model, t_bc, problem.L)
        li = ic_loss(model, x_ic, problem.T_ambient)
        loss = w_pde * lp + w_bc * lb + w_ic * li
        loss.backward()
        return loss

    lbfgs.step(closure)

    rel = relative_l2(model, device)
    print("final relative L2 error vs FDM:", round(rel, 5))

    if not os.path.exists("results"):
        os.makedirs("results")
    torch.save(model.state_dict(), "results/forward_pinn.pth")

    with open("results/forward_metrics.txt", "w") as f:
        f.write("relative_l2 " + str(rel) + "\n")

    return model, rel


if __name__ == "__main__":
    train()
