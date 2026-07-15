import math
import torch
import torch.nn as nn


class HardPINN2D(nn.Module):
    def __init__(self, L, R, T_total, T_ambient, T_scale=40.0, tau=15.0,
                 hidden=64, layers=5, nfreq=8):
        super(HardPINN2D, self).__init__()
        self.L = L
        self.R = R
        self.T_total = T_total
        self.T_ambient = T_ambient
        self.T_scale = T_scale
        self.tau = tau

        ks = []
        for k in range(1, nfreq + 1):
            ks.append(float(k))
        self.register_buffer("ks", torch.tensor(ks).view(1, -1))

        net = []
        net.append(nn.Linear(nfreq + 2, hidden))
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

    def forward(self, x, rho, t):
        feats = torch.cos(math.pi * self.ks * x / self.L)
        tn = t / self.T_total * 2.0 - 1.0
        inp = torch.cat([feats, rho, tn], dim=1)
        out = self.net(inp)
        env = 1.0 - torch.exp(-t / self.tau)
        T = self.T_ambient + self.T_scale * env * out
        return T
