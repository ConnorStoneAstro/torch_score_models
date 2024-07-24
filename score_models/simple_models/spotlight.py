import torch
import torch.nn as nn
from torch.func import grad
from torch import vmap
import numpy as np

from ..ode import RK4_ODE
from .conv_likelihood import PriorNormalScoreModel
from ..score_model import ScoreModel
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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
        self,
        sde,
        priormodel,
        likelihood,
        pilot_samples=None,
        pilot_ll=None,
        N_auto_pilot=1,
        N_live=100,
        K=lambda t: 10,
        epsilon=0.5,
        # use_prodD=True,
    ):
        super().__init__()
        self.sde = sde
        self.priormodel = priormodel
        self.priornormalmodel = ScoreModel(
            sde=sde, model=PriorNormalScoreModel(sde=sde, priormodel=priormodel)
        )
        self.solver = RK4_ODE(self.priornormalmodel)
        self.likelihood = likelihood
        self.pilot_samples = pilot_samples  # P, *D
        self.pilot_ll = pilot_ll  # P, 1
        self.N_auto_pilot = N_auto_pilot
        self.hyperparameters = {"nn_is_energy": True}
        self.K = K
        self.N_live = int(N_live)
        self.live_x0 = None
        self.live_ll = None
        self.epsilon = epsilon
        # self.use_prodD = use_prodD

    def check_convergence(self, ll, scores, sigma_t, xt):
        B, K, *D = scores.shape

        N = torch.argsort(ll, dim=1, descending=True)
        compare = torch.gather(ll, 1, N[:, :2].reshape(B, 2, 1)).reshape(B, 2)
        sample_scores = torch.gather(
            scores, 1, N[:, 1:].reshape(B, K - 1, *[1] * len(D)).expand(B, K - 1, *D)
        )
        sample_ll = torch.gather(ll, 1, N[:, 1:].reshape(B, K - 1, 1))
        sample_ll = torch.exp(sample_ll - torch.max(sample_ll, dim=1, keepdim=True).values)
        sample_ll = sample_ll / torch.sum(sample_ll, dim=1, keepdim=True)
        sample_score = torch.sum(sample_scores * sample_ll, dim=1)

        ll = torch.exp(ll - torch.max(ll, dim=1, keepdim=True).values)
        ll = ll / torch.sum(ll, dim=1, keepdim=True)
        full_score = torch.sum(scores * ll, dim=1)
        print(
            torch.sum(
                torch.linalg.norm(
                    (sample_score - full_score) * sigma_t**2, dim=tuple(range(1, len(D) + 1))
                )
                < (self.epsilon * np.sqrt(np.prod(D)) * sigma_t)
            )
        )
        return torch.all(
            torch.linalg.norm(
                (sample_score - full_score) * sigma_t**2, dim=tuple(range(1, len(D) + 1))
            )
            < (np.sqrt(np.prod(D)) * sigma_t),
        )

    # def check_convergence(self, ll, scores, sigma_t, N_samples=100):
    #     B, K, *D = scores.shape

    #     bootstrap_scores = torch.zeros(B, N_samples, *D, device=scores.device, dtype=scores.dtype)
    #     for n in range(N_samples):
    #         indices = torch.randint(0, K, (B, K))
    #         N = torch.argsort(self.ll, dim=1, descending=True)
    #         sample_scores = torch.gather(
    #             scores, 1, indices.reshape(B, K, *[1] * len(D)).expand(B, K, *D)
    #         )
    #         sample_ll = torch.gather(ll, 1, indices.reshape(B, K, 1))
    #         sample_ll = torch.exp(sample_ll - torch.max(sample_ll, dim=1, keepdim=True).values)
    #         sample_ll = sample_ll / torch.sum(sample_ll, dim=1, keepdim=True)
    #         bootstrap_scores[:, n, :] = torch.sum(sample_scores * sample_ll, dim=1)

    #     sigma_v = torch.std(bootstrap_scores, dim=1)
    #     return torch.all(sigma_v < self.epsilon * sigma_t, dim=1)

    def clear_live_points(self):
        self.live_x0 = None
        self.live_ll = None

    def spotlight_score(self, t, xt, tfloat, sigma_t, live_x0, live_ll):
        K = self.K(tfloat)
        B, *D = xt.shape

        x0 = vmap(
            lambda xf: self.priornormalmodel.sample(
                shape=(K, *xt.shape[1:]), N=50, xf=xf, sigma_f=sigma_t, progress_bar=False
            ),
            randomness="different",
        )(
            xt
        )  # B, K, *D

        ll = vmap(vmap(self.likelihood))(x0)  # B, K, 1
        current_x0 = torch.cat([live_x0, x0], dim=1)  # B, K+L, *D

        current_ll = torch.cat(
            [
                live_ll,
                ll
                + (0.5 / sigma_t**2)
                * torch.sum(
                    (x0 - xt.unsqueeze(1)) ** 2, dim=tuple(range(2, len(x0.shape))), keepdim=True
                ),
                # + torch.log(sigma_t) * np.prod(D),  # fixme why not this term?
            ],
            dim=1,
        )  # B, K+L, 1
        use_x0 = torch.cat(
            [
                current_x0,
                self.pilot_samples.unsqueeze(0).expand(B, *self.pilot_samples.shape),
            ],
            dim=1,
        )  # B, K+L+P, *D
        use_ll = (
            torch.cat(
                [current_ll, self.pilot_ll.unsqueeze(0).expand(B, *self.pilot_ll.shape)], dim=1
            )
            - (0.5 / sigma_t**2)
            * torch.sum(
                (use_x0 - xt.unsqueeze(1)) ** 2,
                dim=tuple(range(2, len(use_x0.shape))),
                keepdim=True,
            )
            # - torch.log(sigma_t) * np.prod(D) # fixme why not this term?
        )  # B, K+L+P, 1
        check = self.check_convergence(use_ll, (use_x0 - xt.unsqueeze(1)) / sigma_t**2, sigma_t, xt)
        use_ll = torch.exp(use_ll - torch.max(use_ll, dim=1, keepdim=True).values)  # B, K+L+P, 1
        w = use_ll / torch.sum(use_ll, dim=(1, 2), keepdim=True)  # B, K+L+P, 1
        w = w.reshape(B, use_x0.shape[1], *[1] * (len(D)))  # B, K+L+P, *[1]*len(D)
        return (
            torch.sum(w * (use_x0 - xt.unsqueeze(1)) / sigma_t, dim=1),
            current_x0,
            current_ll,
            check,
        )

    @torch.no_grad()
    def forward(self, t, xt, **kwargs):
        tfloat = t[0].item()
        sigma_t = self.sde.sigma(t[0])

        # Make initialization of live points if this is first step
        if self.live_x0 is None:
            self.live_x0 = torch.zeros(
                xt.shape[0], 0, *xt.shape[1:], device=xt.device, dtype=xt.dtype
            )
            self.live_ll = torch.zeros(xt.shape[0], 0, 1, device=xt.device, dtype=xt.dtype)
        if self.pilot_samples is None:
            self.pilot_samples = self.priormodel.sample(
                shape=(self.N_auto_pilot, *xt.shape[1:]), N=100, progress_bar=False
            )
            self.pilot_ll = vmap(self.likelihood)(self.pilot_samples)

        # Compute scores
        while True:
            scores, self.live_x0, self.live_ll, check = self.spotlight_score(
                t, xt, tfloat, sigma_t, self.live_x0, self.live_ll
            )

            if self.live_x0.shape[1] > self.N_live:
                self.live_x0 = self.live_x0[:, self.live_x0.shape[1] - self.N_live :]
                self.live_ll = self.live_ll[:, self.live_ll.shape[1] - self.N_live :]
                # N = torch.argsort(self.live_ll, dim=1, descending=False)
                # N = N[:, : self.N_live]
                # B, L, *D = self.live_x0.shape
                # L = self.N_live
                # self.live_x0 = torch.gather(
                #     self.live_x0,
                #     1,
                #     N.reshape(B, L, *[1] * (len(D))).expand(B, L, *D),
                # )
                # self.live_ll = torch.gather(self.live_ll, 1, N)
            if check or True:
                break
        return scores
