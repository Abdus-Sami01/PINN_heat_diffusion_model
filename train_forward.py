import os
import numpy as np
import torch

import problem
from pinn_model import HardPINN
from losses import pde_loss

torch.set_num_threads(4)


def relative_l2(model):
    x_grid, t_grid, T_grid = problem.get_ground_truth()

    xs = []
    ts = []
    truth = []
    for i in range(0, len(x_grid), 5):
        for j in range(0, len(t_grid), 10):
            xs.append(x_grid[i])
            ts.append(t_grid[j])
            truth.append(T_grid[i, j])

    xt = torch.tensor(np.array(xs), dtype=torch.float32).view(-1, 1)
    tt = torch.tensor(np.array(ts), dtype=torch.float32).view(-1, 1)
    truth = np.array(truth)

    model.eval()
    with torch.no_grad():
        pred = model(xt, tt).numpy().flatten()
    model.train()

    return np.linalg.norm(pred - truth) / np.linalg.norm(truth)


def train(adam_epochs=12000, lbfgs_steps=1000, seed=0, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HardPINN(problem.L, problem.T_total, problem.T_ambient,
                     tau=15.0)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.5)

    n_col = 4000

    for epoch in range(adam_epochs):
        x = torch.rand(n_col, 1) * problem.L
        t = torch.rand(n_col, 1) * problem.T_total

        loss = pde_loss(model, x, t, problem.alpha, problem.h_true,
                        problem.Q_torch, problem.T_ambient)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if verbose and epoch % 1000 == 0:
            rel = relative_l2(model)
            print("epoch", epoch, "pde", round(float(loss.item()), 6),
                  "relL2", round(rel, 5), flush=True)

    xf = torch.rand(6000, 1) * problem.L
    tf = torch.rand(6000, 1) * problem.T_total

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = pde_loss(model, xf, tf, problem.alpha, problem.h_true,
                     problem.Q_torch, problem.T_ambient)
        l.backward()
        return l

    lbfgs.step(closure)

    rel = relative_l2(model)
    final_pde = pde_loss(model, xf, tf, problem.alpha, problem.h_true,
                         problem.Q_torch, problem.T_ambient)
    print("final pde loss", round(float(final_pde.item()), 8))
    print("final relative L2 error vs FDM:", round(rel, 5))

    if rel < 0.03:
        print("GATE PASS forward pinn is under 3 percent error")
    else:
        print("GATE FAIL still above 3 percent, do not proceed to phase 3")

    if not os.path.exists("results"):
        os.makedirs("results")
    torch.save(model.state_dict(), "results/forward_pinn.pth")

    with open("results/forward_metrics.txt", "w") as f:
        f.write("relative_l2 " + str(rel) + "\n")
        f.write("final_pde_loss " + str(float(final_pde.item())) + "\n")

    return model, rel


if __name__ == "__main__":
    train()
