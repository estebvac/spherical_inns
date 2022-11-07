import torch.nn as nn
import numpy as np
import torch
from spherical_inns.spherical.sphere_conv import SphereConvBase


class StandardSphericalConvBlock(nn.Module):
    def __init__(self,
                 num_in_channels,
                 num_out_channels,
                 laplacian,
                 depth=2,
                 kernel_size=3,
                 zero_init=False,
                 normalization="instance",
                 activation='LeakyReLU',
                 **kwargs):
        super(StandardSphericalConvBlock, self).__init__()

        conv_op = SphereConvBase

        self.seq = nn.ModuleList()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

        for i in range(depth):

            current_in_channels = max(num_in_channels, num_out_channels)
            current_out_channels = max(num_in_channels, num_out_channels)

            if i == 0:
                current_in_channels = num_in_channels
            if i == depth - 1:
                current_out_channels = num_out_channels

            self.seq.append(
                conv_op(n_feature_in=current_in_channels,
                        n_feature_out=current_out_channels,
                        laplacian=laplacian,
                        kernel_size=kernel_size))

            if normalization == "batch":
                self.seq.append(nn.BatchNorm1d(current_out_channels, eps=1e-3))
            elif normalization == "instance":
                self.seq.append(nn.InstanceNorm1d(current_out_channels))
            elif normalization == "group":
                self.seq.append(nn.GroupNorm(np.min(1, current_out_channels // 8),
                                             current_out_channels,
                                             affine=True))
            else:
                print("No normalization specified.")

            activation = activation.lower()
            if activation == 'softplus':
                self.seq.append(nn.Softplus())
            elif activation == 'relu':
                self.seq.append(nn.ReLU())
            elif activation == 'leakyrelu':
                self.seq.append(nn.LeakyReLU(inplace=True))
            else:
                print("No activation specified.")

        # Initialize the block as the zero transform, such that the coupling
        # becomes the coupling becomes an identity transform (up to permutation
        # of channels)
        if zero_init:
            torch.nn.init.zeros_(self.seq[-1].weight)
            torch.nn.init.zeros_(self.seq[-1].bias)

        self.F = nn.Sequential(*self.seq)

    @classmethod
    def constructor(cls,
                    num_in_channels,
                    num_out_channels,
                    laplacian,
                    **kwargs):
        return cls(num_in_channels,
                   num_out_channels,
                   laplacian,
                   **kwargs)

    def forward(self, x):
        x = self.F(x)
        return x
