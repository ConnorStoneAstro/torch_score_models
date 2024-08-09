import torch
import numpy as np
from .sde import SDE
from torch import Tensor
import numpy as np
from torch.distributions import Normal, Independent
import torch.nn.functional as F
from torch.func import grad, vmap
from score_models.utils import DEVICE


class TSVESDE(SDE):
    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        t_star: float,
        beta: float,
        beta_fn="relu",
        alpha=30,  # silu and hardswish recaling of t
        **kwargs
    ):
        """
        Truncated Scaled Variance Exploding stochastic differential equation

        Args:
            sigma_min (float): The minimum value of the standard deviation of the noise term.
            sigma_max (float): The maximum value of the standard deviation of the noise term.
            t_star (float): Time at which to truncate the VE SDE and start the scaled VE.
            beta (float): Slope of the scale SDE, and also its drift (akin to the VPSDE).
            T (float, optional): The time horizon for the VESDE. Defaults to 1.0.
            device (str, optional): The device to use for computation. Defaults to DEVICE.
        """
        super().__init__(**kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.beta = beta
        self.t_star = t_star
        self.hyperparameters.update(
            {"sigma_min": sigma_min, "sigma_max": sigma_max, "t_star": t_star, "beta": beta}
        )

        if beta_fn == "relu":
            self.beta_fn = lambda t: -self.beta * F.relu(t / self.t_max - self.t_star)
        elif beta_fn == "swish" or beta_fn == "silu":
            self.beta_fn = (
                lambda t: -self.beta * F.silu(alpha * (t / self.t_max - self.t_star)) / alpha
            )
        elif beta_fn == "hardswish":
            self.beta_fn = (
                lambda t: -self.beta * F.hardswish(alpha * (t / self.t_max - self.t_star)) / alpha
            )
        self.beta_fn_dot = vmap(grad(self.beta_fn))

    def sigma(self, t: Tensor) -> Tensor:
        """
        Numerically stable formula for sigma
        """
        smin = np.log(self.sigma_min)
        smax = np.log(self.sigma_max)
        log_coeff = self.beta_fn(t) + (smax - smin) * t / self.t_max + smin
        return torch.exp(log_coeff)

    def mu(self, t: Tensor) -> Tensor:
        """
        Piecewise continuous scale function that takes a VE at t < t_star and
        attach it to a VP-like diffusion at t>t_star. Note that the variance isnan
        still exploding but with a logarihmic slope reduced by the beta hyperparameter.
        """
        return torch.exp(self.beta_fn(t))

    def prior(self, shape, device=DEVICE):
        mu = torch.zeros(shape).to(device)
        sigma_max = np.exp(-self.beta * (1.0 - self.t_star) + np.log(self.sigma_max))
        return Independent(Normal(loc=mu, scale=sigma_max, validate_args=False), len(shape))

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape  # broadcast diffusion coefficient to x shape
        # Analytical derivative of the sigma**2 function, square rooted at the end
        prefactor = np.sqrt(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)))
        return prefactor * self.sigma(t).view(-1, *[1] * len(D))

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape
        return self.beta_fn_dot(t).view(-1, *[1] * len(D)) * x
