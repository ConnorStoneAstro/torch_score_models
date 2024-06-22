from typing import Callable, Union, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch import vmap
from torch.func import grad
import numpy as np
import matplotlib.pyplot as plt


class ConvolvedLikelihood(nn.Module):
    @torch.no_grad()
    def __init__(self, sde, y, Sigma_y, x_shape, A=None, f=None, AAT=None, diag=False):
        assert (A is not None) != (f is not None), "Either A or f must be provided (not both)"
        assert (A is not None) | (AAT is not None), "Either A or AAT must be provided"
        super().__init__()
        self.sde = sde
        self.y = y
        self.Sigma_y = Sigma_y
        self.x_shape = x_shape
        self.y_shape = y.shape
        if A is not None:
            self.A = A.reshape(np.prod(self.y_shape), np.prod(x_shape))
        else:
            self.A = A
        self.f = f
        if AAT is None:
            self.AAT = self.A @ self.A.T
        else:
            self.AAT = AAT
        assert self.Sigma_y.shape == self.AAT.shape, "Sigma_y and AAT must have the same shape"
        self.diag = diag
        self.hyperparameters = {"nn_is_energy": True}

    @property
    def diag(self):
        return self._diag

    @diag.setter
    def diag(self, value):
        self._diag = value
        self.forward = self.diag_forward if value else self.full_forward

    def diag_forward(self, t, xt, **kwargs):
        r = self.y - self.f(xt.squeeze())
        sigma = 1 / (self.Sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * torch.sum(r**2 * sigma).unsqueeze(0)
        return ll.unsqueeze(0) * self.sde.sigma(t)

    def full_forward(self, t, xt, **kwargs):
        r = self.y.reshape(-1) - self.A @ xt.reshape(-1)
        sigma = torch.linalg.inv(self.Sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * (r @ sigma @ r.reshape(1, r.shape[0]).T)
        return ll.unsqueeze(0) * self.sde.sigma(t)


class ConvolvedPriorApproximation(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        sde,
        y: Tensor,
        Sigma_y: Tensor,
        priormodel,
        x_shape: Tuple[int],
        A: Union[Tensor, Callable] = None,
        AAT: Tensor = None,
        ATS1A: Tensor = None,
        A1y: Tensor = None,
        ATS1y: Tensor = None,
        convpriorversion=1,
        gauss_approx_time=0.4,
    ):
        super().__init__()
        self.sde = sde
        self.y = y
        self.y_shape = y.shape
        self.x_shape = x_shape
        self.Sigma_y = Sigma_y
        self.convpriorversion = convpriorversion
        self.gauss_approx_time = gauss_approx_time if convpriorversion == 1 else 2.0
        if isinstance(A, torch.Tensor):
            self.A = A.reshape(np.prod(self.y_shape), np.prod(x_shape))
        else:
            self.A = A
        self.priormodel = priormodel
        if AAT is None:
            self.AAT = self.A @ self.A.T
        else:
            self.AAT = AAT
        if ATS1A is None:
            if Sigma_y.shape == y.shape:
                if self.convpriorversion == 1:
                    ATS1A = torch.sum(self.A**2 / Sigma_y.reshape(-1, 1), dim=0)
                    self.ATS1A = torch.min(ATS1A)
                else:
                    self.ATS1A = self.A.T @ (self.A / Sigma_y.reshape(-1, 1))
            else:
                self.ATS1A = self.A.T @ torch.linalg.inv(Sigma_y) @ self.A
                if self.convpriorversion == 1:
                    self.ATS1A = torch.min(torch.diag(self.ATS1A))
        else:
            self.ATS1A = ATS1A
        print("ATS1y", ATS1y, isinstance(A, torch.Tensor))
        if ATS1y is None and isinstance(A, torch.Tensor):
            if Sigma_y.shape == y.shape:
                self.ATS1y = self.A.T @ (y / Sigma_y).reshape(-1)
            else:
                self.ATS1y = self.A.T @ torch.linalg.inv(Sigma_y) @ y.reshape(-1)
        else:
            self.ATS1y = ATS1y
        if A1y is None:
            self.A1y = (torch.linalg.inv(self.A) @ y.reshape(-1)).reshape(*self.x_shape)
        else:
            self.A1y = A1y
        if Sigma_y.shape == y.shape:
            assert AAT.shape == y.shape, "AAT must have the same shape as y"
        self.hyperparameters = {"nn_is_energy": True}

    @property
    def convpriorversion(self):
        return self._convpriorversion

    @convpriorversion.setter
    def convpriorversion(self, value):
        self._convpriorversion = value
        if value == 1:
            self.prior_score = self.prior_score_v1
        elif value == 2:
            self.prior_score = self.prior_score_v2
        else:
            raise ValueError("convpriorversion must be 1 or 2")

    def conv_like(self, t, xt):
        if isinstance(self.A, torch.Tensor):
            r = self.y - self.A @ xt
        else:
            r = self.y - self.A(xt)
        sigma = self.Sigma_y + self.sde.sigma(t) ** 2 * self.AAT
        if sigma.shape == self.y.shape or sigma.numel() == 1:
            ll = -0.5 * torch.sum(r**2 / sigma)
        else:
            ll = -0.5 * (r @ torch.linalg.inv(sigma) @ r.reshape(1, r.shape[0]).T).squeeze()
        return ll

    def like_score(self, t, xt):
        r = self.A1y - xt
        return r / (1 / self.ATS1A + self.sde.sigma(t[0]) ** 2)

    def prior_score_v1(self, t, xt):
        sigma_t = self.sde.sigma(t[0])
        sigma_c2 = sigma_t**2 / (self.ATS1A * (sigma_t**2 + 1 / self.ATS1A))
        x_c = sigma_c2 * ((self.ATS1A * self.A1y).reshape(1, *self.x_shape) + xt / sigma_t**2)
        t_c = self.sde.t_sigma(torch.sqrt(sigma_c2)) * torch.ones_like(t)
        return self.priormodel.score(t_c, x_c) / (self.ATS1A * (sigma_t**2 + 1 / self.ATS1A))

    def prior_score_v2(self, t, xt):
        B, *D = xt.shape
        sigma_t = self.sde.sigma(t[0])
        Sigma_c = torch.linalg.inv(self.ATS1A + torch.eye(self.ATS1A.shape[0]) / sigma_t**2)
        L = torch.linalg.cholesky(Sigma_c)
        x_c = torch.vmap(torch.matmul, in_dims=(None, 0))(
            Sigma_c, (self.ATS1y + xt / sigma_t**2).reshape(B, -1)
        ).reshape(*xt.shape)
        return torch.vmap(torch.matmul, in_dims=(None, 0))(
            L, self.priormodel.score(t, x_c).reshape(B, -1) / sigma_t
        ).reshape(*xt.shape)

    @torch.no_grad()
    def forward(self, t, xt, **kwargs):

        if t[0].item() > self.gauss_approx_time:
            likelihood_score = self.like_score(t, xt)
        else:
            likelihood_score = vmap(grad(self.conv_like, argnums=1))(t, xt)

        prior_score = self.prior_score(t, xt)
        # print("score compare", torch.mean(likelihood_score), torch.mean(prior_score))
        return (likelihood_score + prior_score) * self.sde.sigma(t[0])


class PriorNormalScoreModel(nn.Module):
    """
    Score model which samples from P(x_0)N(x_0|x_f, sigma_f^2). Where the P(x_0)
    is the prior distribution and may be any distribution, this is represented
    by an SBM which is passed by the user. The user must also supply the inverse
    function t(sigma) which gives the appropriate input time for a given sigma.

    .. math::
        q_t(x_t) = \\int dx_0 P(x_0)N(x_0|x_f, \\sigma_f^2)N(x_t|x_0, \\sigma_t^2)\\\\
        = N(x_t|x_f, \\sigma_f^2 + \\sigma_t^2)\\int dx_0 P(x_0)N(x_0|x_c, \\sigma_c^2)\\\\
        where x_c = \\sigma_c^2(\\sigma_t^{-2} x_t + \\sigma_f^{-2} x_f) and \\sigma_c^2 = (\\sigma_f^{-2} + \\sigma_t^{-2})^{-1}\\\\
        = N(x_t|x_f, \\sigma_f^2 + \\sigma_t^2)P_{t,\\sigma=\\sigma_c}(x_c)
        
    Now we almost have the form of the prior that the SBM would learn, except it
    is in terms of :math:`x_c` rather than :math:`x_t`. We can transform this to
    the the correct formulation analytically as follows:

    .. math::
        \\nabla_{x_t}\\log P_{t,\\sigma=\\sigma_c}(x_c) = \\frac{\\int dx_0 P(x_0)N(x_0|x_c,\\sigma_c^2)\\frac{x_0 - x_c}{\\sigma_c^2}\\frac{\\sigma_c^2}{\\sigma_t^2}}{\\int dx_0 P(x_0)N(x_0|x_c,\\sigma_c^2)} \\\\
        = \\frac{\\sigma_c^2}{\\sigma_t^2}\\nabla_{x_c}\\log P_{t,\\sigma=\\sigma_c}(x_c)

    Therefore we can use the SBM to learn the prior distribution and then transform
    the gradients to the correct form.
    """

    def __init__(self, sde, priormodel, t_sigma):
        super().__init__()
        self.sde = sde
        self.priormodel = priormodel
        self.t_sigma = t_sigma
        self.hyperparameters = {"nn_is_energy": True}

    @torch.no_grad()
    def forward(self, t, xt, xf, sigma_f, **kwargs):
        sigma_t = self.sde.sigma(t[0])
        sigma_c2 = sigma_f**2 * sigma_t**2 / (sigma_f**2 + sigma_t**2)
        xc = (xt * sigma_f**2 + xf * sigma_t**2) / (sigma_f**2 + sigma_t**2)

        Normal_score = (xf - xt) / (sigma_f**2 + sigma_t**2)

        t_c = self.t_sigma(torch.sqrt(sigma_c2)) * torch.ones_like(t)

        Prior_score = self.priormodel(t_c, xc) * sigma_f**2 / (sigma_f**2 + sigma_t**2)

        return (Prior_score + Normal_score) * self.sde.sigma(t[0])
