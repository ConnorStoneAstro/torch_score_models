import torch
from .sde import SDE
from torch import Tensor
import numpy as np


class VESDE(SDE):
    def __init__(
            self,
            sigma_min: float,
            sigma_max: float,
    ):
        """
        Variance Exploding stochastic differential equation 
        
        Args:
            sigma_min (float): The minimum value of the standard deviation of the noise term.
            sigma_max (float): The maximum value of the standard deviation of the noise term.
            T (float, optional): The time horizon for the VESDE. Defaults to 1.0.
            device (str, optional): The device to use for computation. Defaults to DEVICE.
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: Tensor) -> Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    
    def prior(self, dimensions):
        return torch.randn(dimensions) * self.sigma_max
    
    def marginal(self, x0: Tensor, t: Tensor) -> Tensor:
        _, *D = x0.shape
        z = torch.randn_like(x0)
        _, sigma_t = self.marginal_prob_scalars(t)
        return x0 + sigma_t.view(-1, *[1]*len(D)) * z
    
    def marginal_prob_scalars(self, t) -> tuple[Tensor, Tensor]:
        return torch.ones_like(t), self.sigma(t)

    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        _, *D = x.shape # broadcast diffusion coefficient to x shape
        return self.sigma(t).view(-1, *[1]*len(D)) * np.sqrt(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)))

    def drift_f(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


