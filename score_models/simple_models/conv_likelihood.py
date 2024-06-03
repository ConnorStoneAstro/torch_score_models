import torch
import torch.nn as nn
from torch import vmap
from torch.func import grad


class ConvolvedLikelihood(nn.Module):
    def __init__(self, sde, y, sigma_y, A=None, f=None, AAT=None, diag=False):
        assert (A is not None) != (f is not None), "Either A or f must be provided (not both)"
        assert (A is not None) | (AAT is not None), "Either A or AAT must be provided"
        super().__init__()
        self.sde = sde
        self.y = y
        self.sigma_y = sigma_y
        self.A = A
        self.f = f
        if AAT is None:
            self.AAT = A @ A.T
        else:
            self.AAT = AAT
        assert self.sigma_y.shape == self.AAT.shape, "sigma_y and AAT must have the same shape"
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
        sigma = 1 / (self.sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * torch.sum(r**2 * sigma)
        return ll.unsqueeze(0) * self.sde.sigma(t)

    def full_forward(self, t, xt, **kwargs):
        r = self.y - self.A @ xt.squeeze()
        sigma = torch.inverse(self.sigma_y + self.sde.sigma(t) ** 2 * self.AAT)
        ll = 0.5 * (r @ sigma @ r.reshape(1, r.shape[0]).T)
        return ll.unsqueeze(0) * self.sde.sigma(t)


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
