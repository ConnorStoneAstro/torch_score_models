from typing import Callable, Union, Tuple, Optional
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
        sigma = torch.linalg.inv(self.Sigma_y + self.sde.sigma(t[0]) ** 2 * self.AAT)
        ll = 0.5 * (r @ sigma @ r.reshape(1, r.shape[0]).T)
        return ll.unsqueeze(0) * self.sde.sigma(t)


# class ConvolvedPriorApproximation(nn.Module):
#     @torch.no_grad()
#     def __init__(
#         self,
#         sde,
#         y: Tensor,
#         Sigma_y: Tensor,
#         priormodel,
#         x_shape: Tuple[int],
#         A: Union[Tensor, Callable] = None,
#         AAT: Tensor = None,
#         ATS1A: Tensor = None,
#         A1y: Tensor = None,
#         gauss_approx_time=1.1,
#     ):
#         super().__init__()
#         self.sde = sde
#         self.y = y
#         self.y_shape = y.shape
#         self.x_shape = x_shape
#         self.Sigma_y = Sigma_y
#         self.gauss_approx_time = gauss_approx_time
#         if isinstance(A, torch.Tensor):
#             self.A = A.reshape(np.prod(self.y_shape), np.prod(x_shape))
#         else:
#             self.A = A
#         self.priormodel = priormodel
#         if AAT is None:
#             self.AAT = self.A @ self.A.T
#         else:
#             self.AAT = AAT
#         if ATS1A is None:
#             if Sigma_y.shape == y.shape:
#                 ATS1A = torch.sum(self.A**2 / Sigma_y.reshape(-1, 1), dim=0)
#                 self.ATS1A = torch.min(ATS1A)
#             else:
#                 self.ATS1A = self.A.T @ torch.linalg.inv(Sigma_y) @ self.A
#                 self.ATS1A = torch.min(torch.diag(self.ATS1A))
#         else:
#             self.ATS1A = ATS1A
#         if A1y is None:
#             self.A1y = (torch.linalg.inv(self.A) @ y.reshape(-1)).reshape(*self.x_shape)
#         else:
#             self.A1y = A1y
#         if Sigma_y.shape == y.shape:
#             assert (AAT.shape == y.shape) or (AAT.numel() == 1), "AAT must have the same shape as y"
#         self.hyperparameters = {"nn_is_energy": True}

#     def conv_like(self, t, xt):
#         if isinstance(self.A, torch.Tensor):
#             r = self.y - self.A @ xt
#         else:
#             r = self.y - self.A(xt)
#         sigma = self.Sigma_y + self.sde.sigma(t) ** 2 * self.AAT
#         if sigma.shape == self.y.shape or sigma.numel() == 1:
#             ll = -0.5 * torch.sum(r**2 / sigma)
#         else:
#             ll = -0.5 * (r @ torch.linalg.inv(sigma) @ r.reshape(1, r.shape[0]).T).squeeze()
#         return ll

#     def like_score(self, t, xt):
#         r = self.A1y - xt
#         return r / (1 / self.ATS1A + self.sde.sigma(t[0]) ** 2)

#     def prior_score(self, t, xt):
#         sigma_t = self.sde.sigma(t[0])
#         sigma_c2 = sigma_t**2 / (self.ATS1A * sigma_t**2 + 1.0)
#         x_c = sigma_c2 * ((self.ATS1A * self.A1y).reshape(1, *self.x_shape) + xt / sigma_t**2)
#         t_c = self.sde.t_sigma(torch.sqrt(sigma_c2)) * torch.ones_like(t)
#         return self.priormodel.score(t_c, x_c) / (self.ATS1A * sigma_t**2 + 1.0)

#     @torch.no_grad()
#     def forward(self, t, xt, **kwargs):

#         if t[0].item() > self.gauss_approx_time:
#             likelihood_score = self.like_score(t, xt)
#         else:
#             likelihood_score = vmap(grad(self.conv_like, argnums=1))(t, xt)

#         prior_score = self.prior_score(t, xt)
#         # print("score compare", torch.mean(likelihood_score), torch.mean(prior_score))
#         return (likelihood_score + prior_score) * self.sde.sigma(t[0])


# class ConvolvedPriorApproximation(nn.Module):
#     @torch.no_grad()
#     def __init__(
#         self,
#         sde,
#         y: Tensor,
#         Sigma_y: Tensor,
#         priormodel,
#         x_shape: Tuple[int],
#         A: Union[Tensor, Callable] = None,
#         AAT: Tensor = None,
#         ATS1A: Tensor = None,
#         ATS1y: Tensor = None,
#         gauss_approx_time=1.1,
#         gauss_approx_scale=1.0,
#     ):
#         super().__init__()
#         self.sde = sde
#         self.y = y
#         self.y_shape = y.shape
#         self.x_shape = x_shape
#         self.Sigma_y = Sigma_y
#         self.gauss_approx_time = gauss_approx_time
#         self.gauss_approx_scale = gauss_approx_scale
#         if isinstance(A, torch.Tensor):
#             self.A = A.reshape(np.prod(self.y_shape), np.prod(x_shape))
#         else:
#             self.A = A
#         self.priormodel = priormodel
#         if AAT is None:
#             self.AAT = self.A @ self.A.T
#         else:
#             self.AAT = AAT
#         if ATS1A is None:
#             if Sigma_y.shape == y.shape:
#                 ATS1A = torch.sum(self.A**2 / Sigma_y.reshape(-1, 1), dim=0)
#                 self.ATS1A = torch.min(ATS1A)
#             else:
#                 self.ATS1A = self.A.T @ torch.linalg.inv(Sigma_y) @ self.A
#                 self.ATS1A = torch.min(torch.diag(self.ATS1A))
#         else:
#             self.ATS1A = ATS1A
#         if ATS1y is None:
#             if Sigma_y.shape == y.shape:
#                 self.ATS1y = (self.A.T @ (y.reshape(-1) / Sigma_y.reshape(-1))).reshape(
#                     *self.x_shape
#                 )
#             else:
#                 self.ATS1y = (self.A.T @ torch.linalg.inv(Sigma_y) @ y.reshape(-1)).reshape(
#                     *self.x_shape
#                 )
#         else:
#             self.ATS1y = ATS1y

#         self.hyperparameters = {"nn_is_energy": True}

#     def conv_like(self, t, xt):
#         if isinstance(self.A, torch.Tensor):
#             r = self.y - self.A @ xt
#         else:
#             r = self.y - self.A(xt)
#         sigma = self.Sigma_y + self.sde.sigma(t) ** 2 * self.AAT
#         if sigma.shape == self.y.shape or sigma.numel() == 1:
#             ll = -0.5 * torch.sum(r**2 / sigma)
#         else:
#             ll = -0.5 * (r @ torch.linalg.inv(sigma) @ r.reshape(1, r.shape[0]).T).squeeze()
#         return ll

#     def like_score(self, t, xt):
#         r = self.ATS1y * self.gauss_approx_scale - xt
#         return r / (1 / self.ATS1A + self.sde.sigma(t[0]) ** 2)

#     def prior_score(self, t, xt):
#         sigma_t = self.sde.sigma(t[0])
#         sigma_c2 = sigma_t**2 / (self.ATS1A * sigma_t**2 + 1.0)
#         x_c = sigma_c2 * ((self.ATS1y).reshape(1, *self.x_shape) + xt / sigma_t**2)
#         t_c = self.sde.t_sigma(torch.sqrt(sigma_c2)) * torch.ones_like(t)
#         return self.priormodel.score(t_c, x_c) / (self.ATS1A * sigma_t**2 + 1.0)

#     @torch.no_grad()
#     def forward(self, t, xt, **kwargs):

#         if t[0].item() > self.gauss_approx_time:
#             likelihood_score = self.like_score(t, xt)
#         else:
#             likelihood_score = vmap(grad(self.conv_like, argnums=1))(t, xt)

#         prior_score = self.prior_score(t, xt)
#         # print("score compare", torch.mean(likelihood_score), torch.mean(prior_score))
#         return (likelihood_score + prior_score) * self.sde.sigma(t[0])


class ConvolvedPriorApproximation(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        sde,
        y: Tensor,
        Sigma_y: Tensor,
        priormodel,
        x_shape: Tuple[int],
        A: Optional[Tensor] = None,
        f: Optional[Callable] = None,
        l_damp: float = 0.0,
        diag: bool = False,
        AAT: Optional[Tensor] = None,
        ATS1A_full: Optional[Tensor] = None,
        ATS1y: Optional[Tensor] = None,
        A1y: Optional[Tensor] = None,
        gauss_approx_time=1.1,
        gauss_approx_scale=1.0,
    ):
        super().__init__()
        self.sde = sde
        self.y = y
        self.y_shape = y.shape
        self.x_shape = x_shape
        self.Sigma_y = Sigma_y
        self.l_damp = l_damp
        self.gauss_approx_time = gauss_approx_time
        self.gauss_approx_scale = gauss_approx_scale
        self.f = f
        if A is None:
            A = torch.func.jacrev(f)(torch.zeros(x_shape, dtype=y.dtype, device=y.device))
        self.A = A.reshape(np.prod(self.y_shape), np.prod(x_shape))
        self.diag = diag
        self.priormodel = priormodel
        if AAT is None:
            if diag:
                self.AAT = torch.sum(self.A**2, dim=1).reshape(*self.Sigma_y.shape)
            else:
                self.AAT = self.A @ self.A.T
        else:
            self.AAT = AAT

        if ATS1A_full is None:
            if diag:
                self.ATS1A_full = torch.sum(self.A**2 / (Sigma_y.reshape(-1, 1) + l_damp**2), dim=0)
            else:
                self.ATS1A_full = (
                    self.A.T
                    @ torch.linalg.inv(
                        Sigma_y
                        + l_damp**2
                        * torch.eye(Sigma_y.shape[0], dtype=Sigma_y.dtype, device=Sigma_y.device)
                    )
                    @ self.A
                )
        else:
            self.ATS1A_full = ATS1A_full

        if ATS1y is None:
            if Sigma_y.shape == y.shape:
                self.ATS1y = (
                    self.A.T @ (y.reshape(-1) / (Sigma_y.reshape(-1) + l_damp**2))
                ).reshape(*self.x_shape)
            else:
                self.ATS1y = (
                    self.A.T
                    @ torch.linalg.inv(
                        Sigma_y
                        + l_damp**2
                        * torch.eye(Sigma_y.shape[0], dtype=Sigma_y.dtype, device=Sigma_y.device)
                    )
                    @ y.reshape(-1)
                ).reshape(*self.x_shape)
        else:
            self.ATS1y = ATS1y
        if A1y is None:
            if diag:
                self.A1y = self.ATS1y / self.ATS1A_full.reshape(*self.x_shape)
            else:
                self.A1y = (torch.linalg.inv(self.ATS1A_full) @ self.ATS1y.reshape(-1)).reshape(
                    *self.x_shape
                )
        else:
            self.A1y = A1y

        self.hyperparameters = {"nn_is_energy": True}

    def conv_like(self, t, xt, sigma_inv):
        if self.f is None:  # isinstance(self.A, torch.Tensor):
            r = self.y - self.A @ xt
        else:
            r = self.y - self.f(xt)
        if self.diag or sigma_inv.numel() == 1:
            ll = -0.5 * torch.sum(r**2 * sigma_inv)
        else:
            ll = -0.5 * (r.reshape(-1) @ sigma_inv @ r.reshape(1, -1).T).squeeze()
        return ll

    def like_score(self, t, xt):
        r = self.A1y * self.gauss_approx_scale - xt
        if self.diag:
            return r / (1 / self.ATS1A_full.reshape(*self.x_shape) + self.sde.sigma(t[0]) ** 2)
        else:
            return r / (
                1 / torch.diag(self.ATS1A_full).reshape(*self.x_shape) + self.sde.sigma(t[0]) ** 2
            )

    def prior_score(self, t, xt, Sigma_c):
        sigma_t = self.sde.sigma(t)
        if self.diag:
            sigma_c2 = torch.min(Sigma_c)  # sigma_t**2 / (self.ATS1A_scalar * sigma_t**2 + 1.0)
        else:
            sigma_c2 = torch.min(torch.diag(Sigma_c))

        t_c = self.sde.t_sigma(torch.sqrt(sigma_c2)) * torch.ones_like(t)
        if self.diag:
            x_c = Sigma_c * (self.ATS1y + xt / sigma_t**2)
            return (
                Sigma_c
                * self.priormodel.score(t_c.unsqueeze(0), x_c.unsqueeze(0)).squeeze(0)
                / sigma_t**2
            )
        else:
            x_c = (Sigma_c @ ((self.ATS1y).reshape(-1) + xt.reshape(-1) / sigma_t**2)).reshape(
                *self.x_shape
            )
            return (
                Sigma_c
                @ self.priormodel.score(t_c.unsqueeze(0), x_c.unsqueeze(0)).squeeze(0).reshape(-1)
                / sigma_t**2
            ).reshape(*self.x_shape)

    @torch.no_grad()
    def forward(self, t, xt, **kwargs):

        sigma_t = self.sde.sigma(t[0])
        if t[0].item() > self.gauss_approx_time:
            likelihood_score = self.like_score(t, xt)
        else:
            sigma = self.Sigma_y + sigma_t**2 * self.AAT.reshape(*self.Sigma_y.shape)
            if sigma.shape == self.y.shape:
                sigma_inv = 1 / sigma
            else:
                sigma_inv = torch.linalg.inv(sigma)
            likelihood_score = vmap(grad(self.conv_like, argnums=1), in_dims=(0, 0, None))(
                t, xt, sigma_inv
            )

        if self.diag:
            Sigma_c = (1 / (self.ATS1A_full + 1 / sigma_t**2)).reshape(*self.x_shape)
        else:
            Sigma_c = torch.linalg.inv(
                self.ATS1A_full
                + 1
                / sigma_t**2
                * torch.eye(self.ATS1A_full.shape[0], dtype=xt.dtype, device=xt.device)
            )
        # print(
        #     "Sigma_c",
        #     torch.diag(Sigma_c).shape,
        #     torch.diag(Sigma_c).min(),
        #     torch.diag(Sigma_c).max(),
        #     torch.diag(Sigma_c).mean(),
        # )
        prior_score = vmap(self.prior_score, in_dims=(0, 0, None))(t, xt, Sigma_c)
        # print(
        #     "score compare",
        #     torch.mean(likelihood_score),
        #     likelihood_score.abs().max(),
        #     torch.mean(prior_score),
        #     prior_score.abs().max(),
        # )
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

    def __init__(self, sde, priormodel):
        super().__init__()
        self.sde = sde
        self.priormodel = priormodel
        self.hyperparameters = {"nn_is_energy": True}

    @torch.no_grad()
    def forward(self, t, xt, xf, sigma_f, **kwargs):
        sigma_t = self.sde.sigma(t[0])
        sigma_c2 = sigma_f**2 * sigma_t**2 / (sigma_f**2 + sigma_t**2)
        xc = (xt * sigma_f**2 + xf * sigma_t**2) / (sigma_f**2 + sigma_t**2)

        Normal_score = (xf - xt) / (sigma_f**2 + sigma_t**2)

        t_c = self.priormodel.sde.t_sigma(torch.sqrt(sigma_c2)) * torch.ones_like(t)
        Prior_score = self.priormodel.score(t_c, xc) * sigma_c2 / sigma_t**2
        return (Prior_score + Normal_score) * self.priormodel.sde.sigma(t[0])
