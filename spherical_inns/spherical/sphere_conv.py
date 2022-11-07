import torch
import torch.nn as nn
# from deepsphere.layers.chebyshev import SphericalChebConv
from .chebyshev import SphericalChebConv

from .pooling_layers import HealpixPooling


class SphereConvBase(nn.Module):
    """
    Base spherical convolution based on Chebyshev Convolution of deepsphere.
    It adds the handling for permuting the input signal

    Parameters
    ----------
    n_feature_in:   Number of channels of the input signal
    n_feature_out:  Number of channels of the output signal
    laplacian:      Depends on the number of vertex of the input sphere.
    kernel_size:    Select the number of adjacent points to convolve with the signal.
    """

    def __init__(self,
                 n_feature_in,
                 n_feature_out,
                 laplacian,
                 kernel_size):
        super(SphereConvBase, self).__init__()
        self.layers = nn.ModuleList([Permute(),  # Permute since the input is [batch_size, channels, N_vertex]
                                     SphericalChebConv(n_feature_in, n_feature_out, laplacian, kernel_size),
                                     Permute()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SphereConv(nn.Module):
    """
    Spherical convolution with normalization and activation layers.

    Parameters
    ----------
    n_feature_in:   Number of channels of the input signal
    n_feature_out:  Number of channels of the output signal
    laplacian:      Depends on the number of vertex of the input sphere.
    kernel_size:    Select the number of adjacent points to convolve with the signal.
    normalization:  Normalization of the CNN, can be 'batch' or 'instance' normalization

    """

    def __init__(self,
                 n_feature_in,
                 n_feature_out,
                 laplacian,
                 kernel_size,
                 normalization=None,
                 activation=None):
        super(SphereConv, self).__init__()

        convolution = SphereConvBase(n_feature_in, n_feature_out, laplacian, kernel_size)
        normalization_layer = nn.Identity()
        activation_layer = nn.Identity()
        if normalization:
            if normalization == 'batch':
                normalization_layer = nn.BatchNorm1d(n_feature_out)
            elif normalization == 'instance':
                normalization_layer = nn.InstanceNorm1d(n_feature_out)
            else:
                raise NotImplementedError

        if activation:
            if activation == 'softplus':
                activation_layer = nn.Softplus()
            elif activation == 'relu':
                activation_layer = nn.ReLU()

        self.layers = nn.ModuleList([convolution, normalization_layer, activation_layer])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SphereConvBlock(nn.Module):
    """
    Spherical convolution Block with normalization and activation, pooling layers.
    It includes a middle layer to implement double convolution as original U-net implementation.

    Parameters
    ----------
    n_feature_in:   Number of channels of the input signal
    n_feature_out:  Number of channels of the output signal
    laplacian:      Depends on the number of vertex of the input sphere.
    kernel_size:    Select the number of adjacent points to convolve with the signal.
    normalization:  Normalization of the CNN, can be 'batch' or 'instance' normalization
    pooling:        Performs average or max pooling and un-pooling of the signal, laplacian must match with the size
                    after performing the pooling, options are
                    'max':      Max Pooling,
                    'mean':     Average Pooling
                    'un_max':   Max Un-Pooling
                    'un_mean':  Average Un-Pooling

    """

    def __init__(self,
                 n_feature_in,
                 n_feature_out,
                 laplacian,
                 kernel_size,
                 normalization=None,
                 activation=None,
                 pooling=None,
                 n_feature_middle=None):
        super(SphereConvBlock, self).__init__()

        self.pooling_layer = nn.Identity()
        self.pooling_mode = pooling
        if self.pooling_mode:
            self.pooling_layer = HealpixPooling(mode=self.pooling_mode)

        self.middle_layer = nn.Identity()
        if n_feature_middle:
            self.middle_layer = SphereConv(n_feature_in, n_feature_middle, laplacian, kernel_size, normalization,
                                           activation)
            self.convolution = SphereConv(n_feature_middle, n_feature_out, laplacian, kernel_size, normalization,
                                          activation)
        else:
            self.convolution = SphereConv(n_feature_in, n_feature_out, laplacian, kernel_size, normalization,
                                          activation)

    def forward(self, x, index=None, skip_connection=None):
        if self.pooling_mode == 'un_max' and index is None:
            raise AssertionError(f"HealpixMaxUnpool requires indexes from equiangular max pooled sphere")

        indices = None
        if self.pooling_mode:
            x, indices = self.pooling_layer(x, index)

        if skip_connection is not None:
            x = torch.cat((x, skip_connection), dim=1)
        x = self.middle_layer(x)
        x = self.convolution(x)
        return x, indices


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1).contiguous()
