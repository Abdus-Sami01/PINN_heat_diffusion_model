import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, L, T_total, T_ambient, T_scale=40.0,
                 hidden=48, layers=5):
        super(PINN, self).__init__()
        self.L = L
        self.T_total = T_total
        self.T_ambient = T_ambient
        self.T_scale = T_scale

        net = []
        net.append(nn.Linear(2, hidden))
        net.append(nn.Tanh())
        for i in range(layers - 1):
            net.append(nn.Linear(hidden, hidden))
            net.append(nn.Tanh())
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        xn = x / self.L * 2.0 - 1.0
        tn = t / self.T_total * 2.0 - 1.0
        inp = torch.cat([xn, tn], dim=1)
        out = self.net(inp)
        T = self.T_ambient + self.T_scale * out
        return T
