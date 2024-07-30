import torch
import torch.nn as nn
from torch.func import grad
from torch import vmap
import numpy as np

from ..ode import RK4_ODE
from .conv_likelihood import PriorNormalScoreModel
from ..score_model import ScoreModel
from ..solver import EM_SDE
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
        trim=True,
        trim_efolds=100,
    ):
        super().__init__()
        self.sde = sde
        self.priormodel = priormodel
        self.solver = EM_SDE(self.priormodel)
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
        self.trim = trim
        self.trim_efolds = trim_efolds

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
        sample_score = torch.sum(sample_scores * sample_ll.reshape(B, K - 1, *[1] * len(D)), dim=1)

        ll = torch.exp(ll - torch.max(ll, dim=1, keepdim=True).values)
        ll = ll / torch.sum(ll, dim=1, keepdim=True)
        full_score = torch.sum(scores * ll.reshape(B, K, *[1] * len(D)), dim=1)
        # print(
        #     torch.sum(
        #         torch.linalg.norm(
        #             (sample_score - full_score) * sigma_t**2, dim=tuple(range(1, len(D) + 1))
        #         )
        #         < (self.epsilon * np.sqrt(np.prod(D)) * sigma_t)
        #     )
        # )
        return torch.all(
            torch.linalg.norm(
                (sample_score - full_score) * sigma_t**2, dim=tuple(range(1, len(D) + 1))
            )
            < (np.sqrt(np.prod(D)) * sigma_t),
        )

    def clear_live_points(self):
        self.live_x0 = None
        self.live_ll = None
        self.pilot_samples = None
        self.pilot_ll = None

    def spotlight_score(self, t, xt, tfloat, sigma_t, live_x0, live_ll):
        K = self.K(tfloat)
        B, *D = xt.shape

        # Sample P(x0|xt) = P(x0) N(x0|xt,sigma_t^2) / P(xt)
        t_c = self.priormodel.sde.t_sigma(sigma_t).item()
        N = int(100 * t_c + 28)  # fewer steps needed when t_c is small
        x0 = self.solver.reverse(
            xt.unsqueeze(1).expand(B, K, *D).reshape(B * K, *D),
            N=N,
            t_max=t_c,
            progress_bar=False,
        ).reshape(B, K, *D)

        # Likelihood of new samples
        ll = vmap(vmap(self.likelihood))(x0)  # B, K, 1

        # Track all samples so far
        current_x0 = torch.cat([live_x0, x0], dim=1)  # B, K+L, *D

        # Stop logdet term from getting very large when P(x)N(x|xt,sigma_t^2) is basically equal to P(x)
        sigma_t_logdet = torch.min(torch.stack([sigma_t, torch.std(x0)]))

        # Track all likelihoods so far. Include the 1/N(x|xt1,sigma_t1^2) term of the importance weights
        current_ll = torch.cat(
            [
                live_ll,
                ll
                + (0.5 / sigma_t**2)
                * torch.sum(
                    (x0 - xt.unsqueeze(1)) ** 2, dim=tuple(range(2, len(x0.shape)))
                ).unsqueeze(2)
                + torch.log(sigma_t_logdet) * np.prod(D),  # fixme why not this term?
            ],
            dim=1,
        )  # B, K+L, 1

        # Add in the pilot samples, now this is what will be used to compute the scores
        use_x0 = torch.cat([current_x0, self.pilot_samples], dim=1)  # B, K+L+P, *D

        # Add in the pilot sample likelihoods. Also include N(x|xt2,sigma_t2^2) term of the importance sampling
        use_ll = (
            torch.cat([current_ll, self.pilot_ll], dim=1)
            - (0.5 / sigma_t**2)
            * torch.sum(
                (use_x0 - xt.unsqueeze(1)) ** 2, dim=tuple(range(2, len(use_x0.shape)))
            ).unsqueeze(2)
            # - torch.log(sigma_t) * np.prod(D)  # not needed because it is a constant
        )  # B, K+L+P, 1

        # Check if score is unchanged by dropping highest weighted point
        check = self.check_convergence(use_ll, (use_x0 - xt.unsqueeze(1)) / sigma_t**2, sigma_t, xt)

        # Stable compute of weighted exp(log-likelihood)
        use_ll = torch.exp(use_ll - torch.max(use_ll, dim=1, keepdim=True).values)  # B, K+L+P, 1

        # Normalize the weights
        w = use_ll / torch.sum(use_ll, dim=(1, 2), keepdim=True)  # B, K+L+P, 1

        # Reshape to match the samples
        w = w.reshape(B, use_x0.shape[1], *[1] * (len(D)))  # B, K+L+P, *[1]*len(D)

        # Compute the score as weighted average of individual scores. Also
        # Multiply by sigma_t so this may be wrapped in a ScoreModel object
        score = torch.sum(w * (use_x0 - xt.unsqueeze(1)) / sigma_t, dim=1)
        return (
            score,
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

        # Make pilot samples if this is first step
        if self.pilot_samples is None:
            if self.N_auto_pilot <= 0:
                self.pilot_samples = torch.zeros(
                    xt.shape[0], 0, *xt.shape[1:], device=xt.device, dtype=xt.dtype
                )
                self.pilot_ll = torch.zeros(xt.shape[0], 0, 1, device=xt.device, dtype=xt.dtype)
            else:
                self.pilot_samples = vmap(
                    lambda _: self.priormodel.sample(
                        shape=(self.N_auto_pilot, *xt.shape[1:]), N=128, progress_bar=False
                    ),
                    randomness="different",
                )(xt)
                self.pilot_ll = vmap(vmap(self.likelihood))(self.pilot_samples)

        while True:
            # Compute scores
            scores, self.live_x0, self.live_ll, check = self.spotlight_score(
                t, xt, tfloat, sigma_t, self.live_x0, self.live_ll
            )

            # Trim the live points by ultra low likelihood
            if self.live_x0.shape[1] > 5 and self.trim == True:
                # Which points have a likelihood that is reasonably large
                which_good = self.live_ll > (
                    torch.max(self.live_ll, dim=1, keepdim=True).values - self.trim_efolds
                )
                # Find the minimum along each axis that we can cut out
                N_good = torch.sum(which_good, dim=1).detach().cpu().numpy()
                Ncut = self.live_ll.shape[1] - np.max(N_good)
                if Ncut > 0:
                    # Sort the points by likelihood
                    N = torch.argsort(self.live_ll, dim=1, descending=False)
                    N = N[:, Ncut:]

                    # Put the indices back in order
                    N, _ = torch.sort(N, dim=1)

                    # Relevant shapes
                    B, L, *D = self.live_x0.shape
                    L = L - Ncut

                    # Remove the ultra low likelihood points
                    self.live_x0 = torch.gather(
                        self.live_x0,
                        1,
                        N.reshape(B, L, *[1] * (len(D))).expand(B, L, *D),
                    )
                    self.live_ll = torch.gather(self.live_ll, 1, N)

            # Trim the live points by the number of points to user defined maximum
            if self.live_x0.shape[1] > self.N_live:
                self.live_x0 = self.live_x0[:, self.live_x0.shape[1] - self.N_live :]
                self.live_ll = self.live_ll[:, self.live_ll.shape[1] - self.N_live :]
            if check or True:
                break
        return scores
