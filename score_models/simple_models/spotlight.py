import torch
import torch.nn as nn
from torch.func import grad
from torch import vmap
import numpy as np

from ..ode import RK4_ODE
from .conv_likelihood import PriorNormalScoreModel
from ..score_model import ScoreModel


class SpotlightScoreModel(nn.Module):
    """
    Score model for general posterior inference. This model uses a spotlight
    algorithm to sample from the posterior distribution. This is done by
    sampling from the prior distribution and then using the likelihood to weight
    the scores pointing to those samples. The formula for the posterior scores
    can be directly derived as follows:

    .. math::
        \\nabla_{x_t}P_t(x_t|y) = \\frac{\\int dx_0 P(x_0|y)N(x_0|x_t,\\sigma_t^2)\\frac{x_0 - x_t}{\\sigma_t^2}}{\\int dx_0 P(x_0|y)N(x_0|x_t,\\sigma_t^2)}\\\\
        = \\frac{\\mathbb{E}_{P(x_0)N(x_0|x_t,\\sigma_t^2)} \\left[P(y|x_0)\\frac{x_0 - x_t}{\\sigma_t^2}\\right]}{\\mathbb{E}_{P(x_0)N(x_0|x_t,\\sigma_t^2)} \\left[P(y|x_0)\\right]}\\\\
        = \\mathbb{E}_{P(x_0)N(x_0|x_t,\\sigma_t^2)} \\left[w(x_0,x_t)\\frac{x_0 - x_t}{\\sigma_t^2}\\right]

    where the weights are given by: :math:`w(x_0,x_t) = \\frac{P(y|x_0)}{\\mathbb{E}_{P(x_0)N(x_0|x_t,\\sigma_t^2)} \\left[P(y|x_0)\\right]}`.

    To see how to draw the :math:`P(x_0)N(x_0|x_t,\\sigma_t^2)` samples, see the ``PriorNormalScoreModel`` class.
    """

    def __init__(
        self, sde, priormodel, likelihood, t_sigma, N_live=10000, K=lambda t: 10, epsilon=0.1
    ):
        super().__init__()
        self.sde = sde
        self.priornormalmodel = ScoreModel(
            sde=sde, model=PriorNormalScoreModel(sde=sde, priormodel=priormodel, t_sigma=t_sigma)
        )
        self.solver = RK4_ODE(self.priornormalmodel)
        self.likelihood = likelihood
        self.hyperparameters = {"nn_is_energy": True}
        self.K = K
        self.N_live = N_live
        self.live_x0 = None
        self.live_ll = None
        self.epsilon = epsilon

    def check_convergence(self, ll, scores, t_scale, N_samples=100):
        B, K, *D = scores.shape

        bootstrap_scores = torch.zeros(B, N_samples, *D, device=scores.device, dtype=scores.dtype)
        for n in range(N_samples):
            indices = torch.randint(0, K, (B, K))
            sample_scores = torch.gather(
                scores, 1, indices.reshape(B, K, *[1] * len(D)).expand(B, K, *D)
            )
            sample_ll = torch.gather(ll, 1, indices.reshape(B, K, 1))
            sample_ll = torch.exp(sample_ll - torch.max(sample_ll, dim=1, keepdim=True).values)
            sample_ll = sample_ll / torch.sum(sample_ll, dim=1, keepdim=True)
            bootstrap_scores[:, n, :] = torch.sum(sample_scores * sample_ll, dim=1)

        sigma_v = torch.std(bootstrap_scores, dim=1)
        return torch.all(sigma_v < self.epsilon * t_scale, dim=1)

    def spotlight_score(self, t, xt, tfloat, t_scale, live_x0, live_ll):
        NN = int(15 * tfloat + 5)
        K = self.K(tfloat)  # int(100 - 95 * tfloat)
        B, *D = xt.shape
        # epsilon = t_scale * torch.randn(k, *xt.shape, dtype=xt.dtype, device=xt.device)
        # x0 = self.solver.reverse(
        #     xt + epsilon, N=NN, progress_bar=False, t_max=tfloat, xf=xt, sigma_f=t_scale
        # )
        # B, K, *D
        x0 = vmap(
            lambda xf: self.priornormalmodel.sample(
                shape=(K, *xt.shape[1:]), N=100, xf=xf, sigma_f=t_scale, progress_bar=False
            ),
            randomness="different",
        )(
            xt
        )  # B, K, *D
        ll = vmap(vmap(self.likelihood, in_dims=(None, 0)), in_dims=(None, 0))(t, x0)  # B, K, 1
        current_x0 = torch.cat([live_x0, x0], dim=1)  # B, K+L, *D

        current_ll = torch.cat(
            [
                live_ll,
                ll
                + 0.5
                * torch.sum(
                    (x0 - xt.unsqueeze(1)) ** 2 / t_scale**2,
                    dim=tuple(range(2, len(x0.shape))),
                ).unsqueeze(-1)
                + torch.log(t_scale),
            ],
            dim=1,
        )  # B, K+L, 1
        use_ll = (
            current_ll
            - 0.5
            * torch.sum(
                (current_x0 - xt.unsqueeze(1)) ** 2 / t_scale**2, dim=tuple(range(2, len(x0.shape)))
            ).unsqueeze(-1)
            - torch.log(t_scale)
        )  # B, K+L, 1
        print(
            "convergence",
            self.check_convergence(use_ll, (current_x0 - xt.unsqueeze(1)) / t_scale, t_scale),
        )
        use_ll = torch.exp(use_ll - torch.max(use_ll, dim=1, keepdim=True).values)  # B, K+L, 1
        w = use_ll / torch.sum(use_ll, dim=tuple(range(1, len(use_ll.shape)))).reshape(
            B, 1, 1
        )  # B, K+L, 1
        w = w.reshape(B, current_x0.shape[1], *[1] * (len(D)))  # B, K+L, *[1]*len(D)
        return (
            torch.sum(w * (current_x0 - xt.unsqueeze(1)) / t_scale, dim=1),
            current_x0,
            current_ll,
        )

    @torch.no_grad()
    def forward(self, t, xt, **kwargs):
        tfloat = t[0].item()
        t_scale = self.sde.sigma(t[0])

        # Make initialization of live points if this is first step
        if self.live_x0 is None:
            self.live_x0 = torch.zeros(
                xt.shape[0], 0, *xt.shape[1:], device=xt.device, dtype=xt.dtype
            )
            self.live_ll = torch.zeros(xt.shape[0], 0, 1, device=xt.device, dtype=xt.dtype)

        # Compute scores
        scores, self.live_x0, self.live_ll = self.spotlight_score(
            t, xt, tfloat, t_scale, self.live_x0, self.live_ll
        )

        if self.live_x0.shape[1] > self.N_live:
            N = torch.argsort(self.live_ll, dim=1, descending=True)
            N = N[:, : self.N_live]
            B, L, *D = self.live_x0.shape
            L = self.N_live
            self.live_x0 = torch.gather(
                self.live_x0,
                1,
                N.reshape(B, L, *[1] * (len(D))).expand(B, L, *D),
            )
            self.live_ll = torch.gather(self.live_ll, 1, N)

        return scores
