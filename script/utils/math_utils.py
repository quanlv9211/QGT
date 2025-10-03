"""Math utils functions."""

import torch


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5

def inner(u, v, *, keepdim=False, dim=-1):
    r"""
    Minkowski inner product.

    .. math::
        \langle\mathbf{u}, \mathbf{v}\rangle_{\mathcal{L}}:=-u_{0} v_{0}+u_{1} v_{1}+\ldots+u_{d} v_{d}

    Parameters
    ----------
    u : tensor
        vector in ambient space
    v : tensor
        vector in ambient space
    keepdim : bool
        retain the last dim? (default: false)
    dim : int
        reduction dimension

    Returns
    -------
    tensor
        inner product
    """
    return _inner(u, v, keepdim=keepdim, dim=dim)


def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        # return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
        #     dim=dim, keepdim=True
        # )
        return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(
            dim=dim, keepdim=True
        )
