import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

import problem

torch.set_num_threads(4)


class LSTMModel(nn.Module):
    def __init__(self, hidden=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, num_layers=2,
                            batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, seq):
        out, _ = self.lstm(seq)
        return self.head(out)


def build_sequence(x_val, t_vals):
    n = len(t_vals)
    seq = np.zeros((n, 2))
    for i in range(n):
        seq[i, 0] = x_val
        seq[i, 1] = t_vals[i] / problem.T_total
    return seq


def main(n_sensors=3, noise_std=1.0, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    x_grid, t_grid, T_grid = problem.get_ground_truth()

    n_steps = 40
    t_sub = np.linspace(0.0, problem.T_total, n_steps)

    sensor_x = np.linspace(0.15, 0.85, n_sensors)

    seqs = []
    targets = []
    for sx in sensor_x:
        seq = build_sequence(sx, t_sub)
        tgt = np.zeros((n_steps, 1))
        for i in range(n_steps):
            val = problem.interp_T(x_grid, t_grid, T_grid, sx, t_sub[i])
            tgt[i, 0] = val + rng.normal(0.0, noise_std)
        seqs.append(seq)
        targets.append(tgt)

    X = torch.tensor(np.array(seqs), dtype=torch.float32)
    Y = torch.tensor(np.array(targets), dtype=torch.float32)
    Y_mean = Y.mean()
    Y_std = Y.std() + 1e-6

    model = LSTMModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3000):
        pred = model(X)
        loss = torch.mean((pred - (Y - Y_mean) / Y_std) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 1000 == 0:
            print("epoch", epoch, "loss", round(float(loss.item()), 6),
                  flush=True)

    nx = len(x_grid)
    T_pred = np.zeros((nx, n_steps))
    model.eval()
    with torch.no_grad():
        for i in range(nx):
            seq = torch.tensor(build_sequence(x_grid[i], t_sub),
                               dtype=torch.float32).unsqueeze(0)
            out = model(seq).squeeze().numpy()
            T_pred[i, :] = out * float(Y_std) + float(Y_mean)

    T_true = np.zeros((nx, n_steps))
    for i in range(nx):
        for j in range(n_steps):
            T_true[i, j] = problem.interp_T(x_grid, t_grid, T_grid,
                                            x_grid[i], t_sub[j])

    rel = np.linalg.norm(T_pred - T_true) / np.linalg.norm(T_true)
    maxerr = np.abs(T_pred - T_true).max()
    print("LSTM full grid relL2:", round(float(rel), 5))
    print("LSTM max abs error:", round(float(maxerr), 3), "C")

    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/baseline_lstm.txt", "w") as f:
        f.write("relative_l2 " + str(rel) + "\n")
        f.write("max_abs_error " + str(float(maxerr)) + "\n")

    np.save("results/baseline_lstm_grid.npy", T_pred)
    return rel


if __name__ == "__main__":
    main()
