import torch.nn as nn

from .laplacian_funcs import get_healpix_laplacians
from .sphere_conv import SphereConvBlock
import numpy as np
import warnings


class SphericalUnet(nn.Module):
    """
        Spherical U-Net Encoder class, it creates an encoder for the U-Net

        Parameters
        ----------
        encoder_depth:  A number of stages used in encoder in range [3, 5]. Each stage generate features
                        two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
                        with shapes [(B, C, N),], for depth 1 - [(B, C, N), (B, C, N // 2) ] and so on).
                        Default is 5
        normalization:  If **"batch"**, BatchNorm1d layer between SphereConv and Activation layers is used.
                        If **"instance"** InsanceNorm1d will be used.
                        Available options are **"batch", "instance", None**
        enc_channels:   List of integers which specify the number of output channels in each convolution used in
                        encoder. Length of the list should be the same as **encoder_depth**
        in_channels:    Number of channels available in the input of the network.
        pooling:        Type of pooling layers to use after each convolution block. It could be **"mean"** for
                        AveragePooling or **"max"** for MaxPooling.
        kernel_size:    Kernel size used in the Spherical Convolutions of the network.
        """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 subdivision: int,
                 encoder_depth: int = 5,
                 enc_channels: list = None,
                 kernel_size: int = 6,
                 normalization: str = "instance",
                 pooling: str = "max",
                 activation: str = "relu"
                 ):
        super(SphericalUnet, self).__init__()

        if (np.log2(subdivision) + 1) <= encoder_depth:
            warnings.warn(f"Encoder depth adapted from {encoder_depth} to {int(np.log2(subdivision) + 1)}", UserWarning)
            encoder_depth = int(np.log2(subdivision) + 1)

        if enc_channels is None:
            enc_channels = [8, 16, 32, 64, 128]  # [64, 128, 256, 512, 512]
            enc_channels = enc_channels[:encoder_depth]

        self.laplacians = get_healpix_laplacians(nodes=12 * subdivision ** 2,
                                                 depth=encoder_depth,
                                                 laplacian_type="normalized",
                                                 k=8)

        self.encoder = SphericalUnetEncoder(encoder_depth,
                                            enc_channels,
                                            in_channels,
                                            self.laplacians[::-1],
                                            kernel_size,
                                            normalization,
                                            pooling,
                                            activation)

        self.decoder = SphericalUnetDecoder(encoder_depth,
                                            enc_channels,
                                            out_channels,
                                            self.laplacians,
                                            kernel_size,
                                            normalization,
                                            pooling,
                                            activation,
                                            )

    def forward(self, x):
        features, indices = self.encoder(x)
        output = self.decoder(features, indices)
        return output


class SphericalUnetEncoder(nn.Module):
    """
    Spherical U-Net Encoder class, it creates an encoder for the U-Net

    Parameters
    ----------
    encoder_depth:  A number of stages used in encoder in range [3, 5]. Each stage generate features
                    two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
                    with shapes [(B, C, N),], for depth 1 - [(B, C, N), (B, C, N // 2) ] and so on).
                    Default is 5
    normalization:  If **"batch"**, BatchNorm1d layer between SphereConv and Activation layers is used.
                    If **"instance"** InsanceNorm1d will be used.
                    Available options are **"batch", "instance", None**
    enc_channels:   List of integers which specify the number of output channels in each convolution used in
                    encoder. Length of the list should be the same as **encoder_depth**
    in_channels:    Number of channels available in the input of the network.
    pooling:        Type of pooling layers to use after each convolution block. It could be **"mean"** for
                    AveragePooling or **"max"** for MaxPooling.
    laplacians:     List of laplacians at each resolution of the sphere. It must match the resolution for each
                    convolution block.
                    Length of the list should be the same as **encoder_depth**
    kernel_size:    Kernel size used in the Spherical Convolutions of the network.
    """

    def __init__(self,
                 encoder_depth: int,
                 enc_channels: list,
                 in_channels: int,
                 laplacians: list,
                 kernel_size: int,
                 normalization: str = "instance",
                 pooling: str = "max",
                 activation: str = "relu"
                 ):
        super(SphericalUnetEncoder, self).__init__()

        if encoder_depth != len(enc_channels):
            raise ValueError(
                f"Model depth is {encoder_depth}, but you provide `decoder_channels` for {len(enc_channels)} blocks."
            )

        self.enc_channels = enc_channels.copy()
        self.laplacians = laplacians
        self.encoder_depth = encoder_depth
        self.normalization = normalization

        self.enc_channels.insert(0, in_channels)
        self.layers = nn.ModuleList()
        for counter in range(encoder_depth):
            n_feature_middle = None
            n_feature_in = self.enc_channels[counter]

            n_feature_out = self.enc_channels[counter + 1]
            if counter == 0:
                n_feature_middle = self.enc_channels[counter + 1] // 2
            elif counter == encoder_depth - 1:
                n_feature_middle = self.enc_channels[counter + 1]
                n_feature_out = self.enc_channels[counter]

            pooling_mode = None if counter == 0 or counter == encoder_depth else pooling

            lap = self.laplacians[counter]
            print(
                f'in: {n_feature_in},mid: {n_feature_middle}, out: {n_feature_out}, lap: {lap.shape}, pooling: {pooling_mode} ')
            new_layer = SphereConvBlock(n_feature_in=n_feature_in,
                                        n_feature_out=n_feature_out,
                                        laplacian=lap,
                                        kernel_size=kernel_size,
                                        normalization=normalization,
                                        activation=activation,
                                        n_feature_middle=n_feature_middle,
                                        pooling=pooling_mode)
            self.layers.append(new_layer)

        self.pooling = pooling

    def forward(self, x):
        features = list()
        indices = list()
        for layer in self.layers:
            x, index = layer(x)
            features.append(x)
            indices.append(index)
        return features, indices


class SphericalUnetDecoder(nn.Module):
    """
    Spherical U-Net Encoder class, it creates an encoder for the U-Net

    Parameters
    ----------
    decoder_depth:  A number of stages used in encoder in range [3, 5]. Each stage generate features
                    two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
                    with shapes [(B, C, N),], for depth 1 - [(B, C, N), (B, C, N // 2) ] and so on).
                    Default is 5
    normalization:  If **"batch"**, BatchNorm1d layer between SphereConv and Activation layers is used.
                    If **"instance"** InsanceNorm1d will be used.
                    Available options are **"batch", "instance", None**
    decoder_channels:   List of integers which specify the number of output channels in each convolution used in
                    encoder. Length of the list should be the same as **encoder_depth**
    out_channels:    Number of channels available in the input of the network.
    pooling:        Type of pooling layers to use after each convolution block. It could be **"mean"** for
                    AveragePooling or **"max"** for MaxPooling.
    laplacians:     List of laplacians at each resolution of the sphere. It must match the resolution for each
                    convolution block.
                    Length of the list should be the same as **encoder_depth**
    kernel_size:    Kernel size used in the Spherical Convolutions of the network.
    """

    def __init__(self,
                 decoder_depth: int,
                 decoder_channels: list,
                 out_channels: int,
                 laplacians: list,
                 kernel_size: int,
                 normalization: str = "instance",
                 pooling: str = "max",
                 activation: str = "relu"):
        super(SphericalUnetDecoder, self).__init__()
        if decoder_depth != len(decoder_channels):
            raise ValueError(
                f"Model depth is {decoder_depth}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."
            )

        if pooling == 'max' or pooling == 'mean':
            pooling = 'un_' + pooling

        self.decoder_channels = decoder_channels.copy()[::-1][1:]
        self.laplacians = laplacians
        self.decoder_depth = decoder_depth - 1
        self.normalization = normalization

        self.layers = nn.ModuleList()
        for counter in range(self.decoder_depth):
            pooling_mode = None if counter == self.decoder_depth else pooling
            n_feature_in = self.decoder_channels[counter] if counter == self.decoder_depth \
                else self.decoder_channels[counter] * 2
            n_feature_middle = self.decoder_channels[counter]
            n_feature_out = out_channels if counter == self.decoder_depth - 1 else self.decoder_channels[counter + 1]
            lap = self.laplacians[min(counter + 1, decoder_depth - 1)]
            # print(f'in: {n_feature_in},mid: {n_feature_middle}, out: {n_feature_out}, lap: {lap.shape}, pooling: {
            # pooling_mode} ')
            new_layer = SphereConvBlock(n_feature_in=n_feature_in,
                                        n_feature_out=n_feature_out,
                                        laplacian=lap,
                                        kernel_size=kernel_size,
                                        normalization=normalization,
                                        activation=activation,
                                        n_feature_middle=n_feature_middle,
                                        pooling=pooling_mode)
            self.layers.append(new_layer)

        self.pooling = pooling

    def forward(self, feats, indexes):
        x = feats[-1]
        features = feats[:-1][::-1]
        indices = [ix for ix in indexes if ix is not None][::-1]

        for counter, layer in enumerate(self.layers):
            skip_connection = None if counter == self.decoder_depth else features[counter]
            index = None if counter == self.decoder_depth else indices[counter]
            x, index = layer(x, index=index, skip_connection=skip_connection)

        return x
