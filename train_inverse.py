import os
import numpy as np
import torch

import problem
from pinn_model import HardPINN
from losses import pde_loss, data_loss

torch.set_num_threads(4)


def train_inverse(n_sensors=3, n_times=8, noise_std=1.0, h_init=0.15,
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

    h_raw = torch.nn.Parameter(torch.tensor(float(np.log(np.exp(h_init) - 1.0))))

    params = list(model.parameters()) + [h_raw]
    opt = torch.optim.Adam(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.5)

    w_data = 10.0
    n_col = 4000

    h_history = []

    for epoch in range(adam_epochs):
        h = torch.nn.functional.softplus(h_raw)

        x = torch.rand(n_col, 1) * problem.L
        t = torch.rand(n_col, 1) * problem.T_total

        lp = pde_loss(model, x, t, problem.alpha, h,
                      problem.Q_torch, problem.T_ambient)
        ld = data_loss(model, x_obs, t_obs, T_obs)

        loss = lp + w_data * ld

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 50 == 0:
            h_history.append(float(h.item()))

        if verbose and epoch % 1000 == 0:
            print("epoch", epoch, "pde", round(float(lp.item()), 6),
                  "data", round(float(ld.item()), 6),
                  "h", round(float(h.item()), 5), flush=True)

    xf = torch.rand(6000, 1) * problem.L
    tf = torch.rand(6000, 1) * problem.T_total

    lbfgs = torch.optim.LBFGS(params, max_iter=lbfgs_steps,
                              history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        h = torch.nn.functional.softplus(h_raw)
        lp = pde_loss(model, xf, tf, problem.alpha, h,
                      problem.Q_torch, problem.T_ambient)
        ld = data_loss(model, x_obs, t_obs, T_obs)
        l = lp + w_data * ld
        l.backward()
        return l

    lbfgs.step(closure)

    h_final = float(torch.nn.functional.softplus(h_raw).item())
    h_err = abs(h_final - problem.h_true) / problem.h_true * 100.0

    h_history.append(h_final)

    print("recovered h:", round(h_final, 5))
    print("true h:", problem.h_true)
    print("relative error:", round(h_err, 2), "percent")

    return model, h_final, h_err, h_history


if __name__ == "__main__":
    model, h_final, h_err, h_history = train_inverse()

    if not os.path.exists("results"):
        os.makedirs("results")
    torch.save(model.state_dict(), "results/inverse_pinn.pth")
    with open("results/inverse_metrics.txt", "w") as f:
        f.write("h_recovered " + str(h_final) + "\n")
        f.write("h_true " + str(problem.h_true) + "\n")
        f.write("h_rel_error_pct " + str(h_err) + "\n")
    np.save("results/h_history.npy", np.array(h_history))
