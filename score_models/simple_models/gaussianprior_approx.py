from typing import Callable, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch import vmap
from torch.func import grad
import numpy as np
from .conv_likelihood import ConvolvedPriorApproximation


class GaussianPriorApproximation(nn.Module):
    def __init__(
        self,
        sde,
        y: Tensor,
        Sigma_y: Tensor,
        priormodel,
        xp: Tensor,
        Sigma_p: Tensor,
        A: Union[Tensor, Callable] = None,
        f: Callable = None,
        AAT: Tensor = None,
    ):
        super().__init__()
        self.sde = sde
        self.y = y
        self.Sigma_y = Sigma_y
        self.priormodel = priormodel
        self.xp = xp
        self.Sigma_p = Sigma_p
        if isinstance(A, torch.Tensor):
            self.A = A.reshape(np.prod(self.y.shape), np.prod(self.xp.shape))
        else:
            self.A = A
        self.f = f
        if AAT is None:
            self.AAT = self.A @ self.A.T
        else:
            self.AAT = AAT

        if Sigma_y.shape == y.shape:
            assert AAT.shape == y.shape, "AAT must have the same shape as y"
        self.hyperparameters = {"nn_is_energy": True}

    def conv_like(self, t, xt):

        if self.Sigma_p.shape == self.xp.shape or self.Sigma_p.numel() == 1:
            Sigma_c = (
                self.Sigma_p * self.sde.sigma(t) ** 2 / (self.Sigma_p + self.sde.sigma(t) ** 2)
            )
            Sigma_c = torch.max(Sigma_c)
            x_c = Sigma_c * (self.xp / self.Sigma_p + xt / self.sde.sigma(t) ** 2)
            sigma = self.Sigma_y + Sigma_c * self.AAT
            # sigma = self.Sigma_y + torch.sum(self.A**2 * Sigma_c.reshape(1, -1), dim=1).reshape(
            #     self.y.shape
            # )
        else:
            Sigma_c = torch.linalg.inv(
                torch.linalg.inv(self.Sigma_p)
                + 1 / self.sde.sigma(t) ** 2 * torch.eye(self.Sigma_p.shape[0])
            )
            x_c = Sigma_c @ (torch.linalg.inv(self.Sigma_p) @ self.xp + xt / self.sde.sigma(t) ** 2)
            sigma = self.Sigma_y + self.A @ Sigma_c @ self.A.T  # fixme all cases
        if self.f is None:
            r = self.y - self.A @ x_c
        else:
            r = self.y - self.f(x_c)

        if sigma.shape == self.y.shape:
            ll = -0.5 * torch.sum(r**2 / sigma)
        else:
            ll = -0.5 * (r @ torch.linalg.inv(sigma) @ r.reshape(1, r.shape[0]).T).squeeze()
        return ll

    def forward(self, t, xt, **kwargs):

        likelihood_score = vmap(grad(self.conv_like, argnums=1))(t, xt)

        prior_score = self.priormodel.score(t, xt)
        # print("score compare", torch.mean(likelihood_score), torch.mean(prior_score))
        return (likelihood_score + prior_score) * self.sde.sigma(t[0])
