"""Original implementation in:
https://github.com/AxelElaldi/equivariant-spherical-deconvolution/blob/a9542f65991fb6e8b1f05aaaaf2f663cc0f43e31/model/utils_dataset.py#L456
"""

import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    def __init__(self, norm, sigma=None):
        """
        Parameters
        ----------
        norm : str
            Name of the loss.
        sigma : float
            Hyper parameter of the loss.
        """
        super(WeightedLoss, self).__init__()
        if norm not in ['L2', 'L1', 'cauchy', 'welsch', 'geman']:
            raise NotImplementedError('Expected L1, L2, cauchy, welsh, geman but got {}'.format(norm))
        if sigma is None and norm in ['cauchy', 'welsch', 'geman']:
            raise NotImplementedError('Expected a loss hyper parameter for {}'.format(norm))
        self.norm = norm
        self.sigma = sigma

    def forward(self, img1, img2, wts=None, mask=None):
        """
        Parameters
        ----------
        img1 : torch.Tensor
            Prediction tensor
        img2 : torch.Tensor
            Ground truth tensor
        wts: torch.nn.Parameter
            If specified, the weight of the grid.
        mask: torch.nn.Parameter
            If specified, which values use to compute the loss.
        Returns
        -------
         loss : torch.Tensor
            Loss of the predicted tensor
        """
        if self.norm == 'L2':
            out = (img1 - img2) ** 2
        elif self.norm == 'L1':
            out = torch.abs(img1 - img2)
        elif self.norm == 'cauchy':
            out = 2 * torch.log(1 + ((img1 - img2) ** 2 / (2 * self.sigma)))
        elif self.norm == 'welsch':
            out = 2 * (1 - torch.exp(-0.5 * ((img1 - img2) ** 2 / self.sigma)))
        elif self.norm == 'geman':
            out = 2 * (2 * ((img1 - img2) ** 2 / self.sigma) / ((img1 - img2) ** 2 / self.sigma + 4))
        else:
            raise ValueError('Expected L1, L2, cauchy, welsh, geman but got {}'.format(self.norm))

        if wts is not None:
            out = out * wts

        if mask is not None:
            loss = out[mask].sum() / mask.sum()
        else:
            loss = out.mean()  # out.sum() / (out.size(0) * out.size(1))
        return loss


class NonZeroLoss(nn.Module):
    def __init__(self, sigma=None):
        """
        Parameters
        ----------
        norm : str
            Name of the loss.
        sigma : float
            Hyper parameter of the loss.
        """
        super(NonZeroLoss, self).__init__()
        self.sigma = sigma

    def forward(self, img1, dim, wts=None):
        # print(img1.shape)
        # out = torch.exp(-torch.norm(img1, dim=dim, keepdim=True) * (self.sigma ** -1))
        out = 1 / (torch.norm(img1, dim=dim, keepdim=True) * (self.sigma ** -1))
        out = torch.clamp(out, min=0., max=50.)
        # print(out.shape)
        if wts is not None:
            out = out * wts
            wts = torch.ones_like(out) * wts
            loss = out.sum() / wts.sum()
        else:
            loss = out.mean()

        return loss


class RadialDecayLoss(nn.Module):
    def __init__(self, grid, sigma: float = 0.1):
        """
        Parameters
        ----------
        norm : str
            Name of the loss.
        sigma : float
            Hyper parameter of the loss.
        """
        super(RadialDecayLoss, self).__init__()
        self.grid = torch.tensor(grid)
        self.sigma = sigma

    def forward(self, odf, center=0, max_theta=None):
        # print(img1.shape)
        # out = torch.exp(-torch.norm(img1, dim=dim, keepdim=True) * (self.sigma ** -1))
        if self.grid.dtype != odf.dtype:
            self.grid = self.grid.to(odf.device).to(odf.dtype)
        error = torch.abs(self.grid - center)
        radial = torch.exp(-error / self.sigma)
        radial = torch.where(radial < 1e-4, torch.zeros_like(radial), radial)
        out = torch.abs(odf) * radial[None, None, :]

        if max_theta is not None:
            mask = self.grid <= max_theta
            mask = mask[None, None, :]
            out = out * mask
            loss = out.sum() / mask.sum()
        else:
            loss = out.mean()

        return loss
