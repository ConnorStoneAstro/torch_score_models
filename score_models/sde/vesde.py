import torch
from .sde import SDE
from torch import Tensor
import numpy as np
from score_models.utils import DEVICE


class VESDE(SDE):
    def __init__(self, sigma_min: float, sigma_max: float, **kwargs):
        """
        Variance Exploding stochastic differential equation

        Args:
            sigma_min (float): The minimum value of the standard deviation of the noise term.
            sigma_max (float): The maximum value of the standard deviation of the noise term.
            T (float, optional): The time horizon for the VESDE. Defaults to 1.0.
            device (str, optional): The device to use for computation. Defaults to DEVICE.
        """
        super().__init__(**kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t / self.t_max)

    def t_sigma(self, sigma: Tensor) -> Tensor:
        return torch.log(torch.as_tensor(sigma / self.sigma_min, device=DEVICE)) / torch.log(
            torch.as_tensor(self.sigma_max / self.sigma_min, device=DEVICE)
        )

    def mu(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape  # broadcast diffusion coefficient to x shape
        return self.sigma(t).view(-1, *[1] * len(D)) * np.sqrt(
            2 * (np.log(self.sigma_max) - np.log(self.sigma_min))
        )

    def drift(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x)
