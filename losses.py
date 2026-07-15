import torch


def pde_residual(model, x, t, alpha, h, Q_func, T_ambient):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    T = model(x, t)

    T_t = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T),
                              create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                              create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, grad_outputs=torch.ones_like(T_x),
                               create_graph=True)[0]

    Q = Q_func(x, t)
    res = T_t - alpha * T_xx + h * (T - T_ambient) - Q
    return res


def pde_loss(model, x, t, alpha, h, Q_func, T_ambient):
    res = pde_residual(model, x, t, alpha, h, Q_func, T_ambient)
    return torch.mean(res ** 2)


def bc_loss(model, t_bc, L):
    zeros = torch.zeros_like(t_bc)
    ends = torch.full_like(t_bc, L)

    x0 = zeros.clone().detach().requires_grad_(True)
    T0 = model(x0, t_bc)
    T0_x = torch.autograd.grad(T0, x0, grad_outputs=torch.ones_like(T0),
                               create_graph=True)[0]

    xL = ends.clone().detach().requires_grad_(True)
    TL = model(xL, t_bc)
    TL_x = torch.autograd.grad(TL, xL, grad_outputs=torch.ones_like(TL),
                               create_graph=True)[0]

    return torch.mean(T0_x ** 2) + torch.mean(TL_x ** 2)


def ic_loss(model, x_ic, T_ambient):
    t0 = torch.zeros_like(x_ic)
    T = model(x_ic, t0)
    target = torch.full_like(T, T_ambient)
    return torch.mean((T - target) ** 2)


def data_loss(model, x_obs, t_obs, T_obs):
    T = model(x_obs, t_obs)
    return torch.mean((T - T_obs) ** 2)


def grad_norm(loss, model):
    grads = torch.autograd.grad(loss, model.parameters(),
                                retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total = total + torch.sum(g ** 2)
    return torch.sqrt(total)
