import torch
import torch.nn as nn
import numpy as np


class AnnealedScoreModel(nn.Module):
    """
    Smoothly transitions between two score models as a function of t.

    This score model class allows for the interpolation of the scores between
    two models. Can be useful when one model is better at capturing the score in
    the early stages of the SDE and another model is better at capturing the
    score in the later stages of the SDE.

    Args:
        sde: The SDE that the score model is associated with.
        hight_model: The high temperature model.
        lowt_model: The low temperature model.
        beta_scheme: The scheme for the beta parameter. Can be "linear", "square",
            "sqrt", "linear:<i>", "sqrt:<i>", or "sin:<i>". For the "<i>" models
            the ``i`` parameter can be used to scale the ``t`` input to beta making
            the transition happen later.

    """

    def __init__(self, sde, hight_model, lowt_model, beta_scheme="linear", **kwargs):
        super().__init__()
        self.sde = sde
        self.hight_model = hight_model
        self.lowt_model = lowt_model
        self.beta_scheme = beta_scheme

    def beta(self, t):
        T = (t - self.sde.t_min) / (self.sde.t_max - self.sde.t_min)
        if self.beta_scheme == "linear":
            return T
        elif self.beta_scheme == "square":
            return T**2
        elif self.beta_scheme == "sqrt":
            return torch.sqrt(T)
        elif "linear:" in self.beta_scheme:
            return int(self.beta_scheme[self.beta_scheme.find(":") + 1 :]) * T
        elif "sqrt:" in self.beta_scheme:
            return torch.sqrt(int(self.beta_scheme[self.beta_scheme.find(":") + 1 :]) * T)
        elif "sin:" in self.beta_scheme:
            i = int(self.beta_scheme[self.beta_scheme.find(":") + 1 :])
            return torch.where(T > 1 / i, torch.ones_like(T), torch.sin(i * T * torch.pi / 2.0))
        else:
            raise NotImplementedError(f"Unknown beta_scheme {self.beta_scheme}")

    def forward(self, t, x, **kwargs):
        B, *D = x.shape
        # Compute the weighted score for each model
        beta = torch.clamp(self.beta(kwargs.get("t_a", t)[0]), 0.0, 1.0)
        score = self.hight_model(t, x, **kwargs) * beta
        score += self.lowt_model(t, x, **kwargs) * (1.0 - beta)
        return score * self.sde.sigma(t).view(-1, *[1] * len(D))
