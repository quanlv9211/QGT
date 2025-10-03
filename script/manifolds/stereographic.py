import torch

from geoopt.manifolds.stereographic import math as gmath


class StereographicModel():
    def __init__(self, **kwargs):
        super().__init__()
        self.name = 'StereographicModel'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def dist(self, x, y, c):
        return self.sqdist(x, y, c).sqrt()

    def sqdist(self, x, y, c_):
        x, c = self.split(x, c_)
        y, _ = self.split(y, c_)
        # x: (*, P, D), y: (*, P, D)
        
        x2 = x.pow(2).sum(dim=-1, keepdim=True)  # (*, P, 1)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)  # (*, P, 1)
        # xy = torch.matmul(y, (-x).transpose(-2, -1))  # (N, N, 1)
        xy = torch.matmul(y.unsqueeze(-2), (-x).unsqueeze(-1)).squeeze(-2)  # (*, P, 1)

        denom = 1 - 2 * c * xy + c.pow(2) * x2 * y2
        a = (1 - 2 * c * xy - c * y2) / denom
        b = (1 + c * x2) / denom

        # mobius_add(x, y) = -ax + by
        norm = a.pow(2) * x2 + 2 * a * b * xy + b.pow(2) * y2
        dist = 2.0 * gmath.artan_k(
            norm.clip(min=1e-15).sqrt(), k=c
        )  # (*, P, 1)
        
        return dist.pow(2).sum(dim=-2)

    def proj(self, x, c):
        x, c = self.split(x, c)
        return self.merge(gmath.project(x, k=c))

    def expmap(self, u, p, c_):
        u, c = self.split(u, c_)
        p, _ = self.split(p, c_)
        
        return self.merge(gmath.expmap(u, p, k=c))

    def logmap(self, p1, p2, c_):
        p1, c = self.split(p1, c_)
        p2, _ = self.split(p2, c_)
        return self.merge(gmath.logmap(p1, p2, k=c))

    def expmap0(self, u, c):
        u, c = self.split(u, c)
        return self.merge(gmath.expmap0(u, k=c))

    def logmap0(self, p, c):
        p, c = self.split(p, c)
        return self.merge(gmath.logmap0(p, k=c))

    def mobius_add(self, x, y, c_):
        x, c = self.split(x, c_)
        y, _ = self.split(y, c_)
        return self.merge(gmath.mobius_add(x, y, k=c))
    
    def mobius_sub(self, x, y, c_):
        x, c = self.split(x, c_)
        y, _ = self.split(y, c_)
        return self.merge(gmath.mobius_add(x, -y, k=c))

    def inner(self, v1, v2, keepdim=False):
        return (v1 * v2).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c_):
        x, c = self.split(x, c_)
        y, _ = self.split(y, c_)
        u, _ = self.split(u, c_)
        return self.merge(gmath.parallel_transport(x, y, u, k=c))

    def ptransp0(self, x, u, c_):
        x, c = self.split(x, c_)
        u, _ = self.split(u, c_)
        return self.merge(gmath.parallel_transport0(x, u, k=c))

    def lambda_x(self, x, c, dim=-1, keepdim=False):
        x, c = self.split(x, c)
        return self.merge(gmath.lambda_x(x, k=c, keepdim=keepdim, dim=dim))

    def mobius_scalar_mul(self, a, x, c_):
        if len(a.shape) != 0:
            a = a.unsqueeze(-1)
        x, c = self.split(x, c_)
        return self.merge(gmath.mobius_scalar_mul(a, x, k=c))

    def clamp_abs(self, x, eps):
        return gmath.clamp_abs(x, eps)

    def antipode(self, x, c):
        x, c = self.split(x, c)
        return self.merge(gmath.antipode(x, k=c))

    def arsin_k(self, x, c):
        x, c = self.split(x, c)
        return self.merge(gmath.arsin_k(x, k=c))

    def split(self, x, c):
        np = c.shape[0]
        D = x.shape[-1]
        others = x.shape[:-1]
        x = x.reshape(*others, np, D // np)
        c = c.unsqueeze(-1)
        while len(c.shape) < len(x.shape):
            c = c.unsqueeze(0)
        return x, c

    def merge(self, x):
        P, D = x.shape[-2:]
        others = x.shape[:-2]
        x = x.reshape(*others, P * D)
        return x

