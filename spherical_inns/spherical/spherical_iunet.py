from typing import Iterable
from typing import Union, Any, Tuple, List, Optional

import numpy as np
import torch
from torch import Tensor
import warnings

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import GraphINN
from FrEIA.framework import SequenceINN
from spherical_inns.spherical.spherical_ilayer import StandardSphericalConvBlock
from FrEIA.iunets.utils import print_iunet_layout
from .laplacian_funcs import get_healpix_laplacians

coupling_layers = {'nice': Fm.NICECouplingBlock,
                   'all': Fm.AllInOneBlock,
                   'nvp': Fm.RNVPCouplingBlock,
                   'glow': Fm.GLOWCouplingBlock,
                   'gin': Fm.GINCouplingBlock,
                   'affine': Fm.AffineCouplingOneSided,
                   }

from FrEIA.iunets.layers import (InvertibleDownsampling1D,
                                 InvertibleUpsampling1D)


class IUnetCOnvBlock(SequenceINN):

    def __init__(self,
                 dims_in,
                 laplacian,
                 depth: int = 2,
                 n_conv: int = 2,
                 normalization: str = "instance",
                 padding_mode='zeros',
                 coupling_layer='glow',
                 inv_block_kwargs: dict = {},
                 **kwargs):
        if isinstance(dims_in, list):
            dims_in = dims_in[0]

        coupling_fc = coupling_layers[coupling_layer]
        super().__init__(*dims_in)
        kwargs = {"depth": n_conv,
                  "zero_init": False,
                  "normalization": normalization,
                  "padding_mode": padding_mode, }
        subnet_constructor = lambda ch_in, ch_out: StandardSphericalConvBlock.constructor(ch_in,
                                                                                          ch_out,
                                                                                          laplacian,
                                                                                          **kwargs)
        for n_rep in range(depth):
            self.append(Fm.PermuteRandom)
            self.append(coupling_fc, subnet_constructor=subnet_constructor, **inv_block_kwargs)

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        return input_dims

    def forward(self, x_or_z: Tensor, c: Iterable[Tensor] = None,
                rev: bool = False, jac: bool = True) -> Tuple[Tensor, Tensor]:

        x_out, log_jac_det = super().forward(x_or_z, c, rev, jac)
        return (x_out,), log_jac_det


class SphericaliUNet(GraphINN):
    def __init__(self,
                 in_channels: int,  # used
                 subdivision: int,
                 channels: Union[Tuple[int, ...], List[Optional[int]]],  # used
                 architecture: Union[Tuple[int, ...], List[Optional[int]]],  # used
                 conv_depth: Union[int, Tuple[int, ...], List[Optional[int]]] = 2,
                 module_kwargs: dict = None,
                 learnable_resampling: bool = True,  # used
                 resampling_method: str = "cayley",  # used
                 resampling_init: Union[str, np.ndarray, torch.Tensor] = "haar",  # used
                 resampling_kwargs: dict = None,  # used
                 coupling_layer: str = 'affine',
                 # permute_soft: bool = False,
                 learned_householder_permutation: int = 0,
                 reverse_permutation: bool = False,
                 verbose: int = 1,  # used
                 normalization_conv: Optional[str] = 'instance',  # used
                 padding_mode: Union[str, type(None)] = "replicate",  # used
                 **kwargs: Any):

        if (np.log2(subdivision) + 1) <= len(architecture):
            warnings.warn(f"architecture and channels adapted from len {len(architecture)} to"
                          f" len  {int(np.log2(subdivision) + 1)}", UserWarning)
            architecture = architecture[:int(np.log2(subdivision) + 1)]
            channels = channels[:int(np.log2(subdivision) + 1)]

        self.laplacians = get_healpix_laplacians(nodes=12 * subdivision ** 2,
                                                 depth=len(architecture),
                                                 laplacian_type="normalized",
                                                 k=8)
        self.laplacians = [lap for lap in reversed(self.laplacians)]

        channels = self.check_channels(channels)
        self.in_channels = in_channels
        self.architecture = architecture
        if isinstance(conv_depth, int):
            self.conv_depth = (conv_depth,) * len(self.architecture)
        elif isinstance(conv_depth, (tuple, list)):
            self.conv_depth = conv_depth
        else:
            TypeError("conv_depth must be a int, tuple or list")

        self.dim = 1
        self.num_levels = len(architecture)
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = module_kwargs

        if len(channels) != len(self.architecture):
            raise AttributeError("channels must have the same length as architecture.")
        if len(self.architecture) != len(self.conv_depth):
            raise AttributeError("conv_depth must have the same length as architecture.")

        self.channels = [channels[0]]
        self.channels_before_downsampling = []
        self.skipped_channels = []

        # --- Padding attributes ---
        self.padding_mode = padding_mode

        # --- AllInOneBlock attributes ---
        inv_block_kwargs = {}  # dict(permute_soft=permute_soft,
        # learned_householder_permutation=learned_householder_permutation,
        # reverse_permutation=reverse_permutation)

        # --- Invertible up- and downsampling attributes ---
        # Reformat resampling_stride to the standard format
        resampling_stride = 4  # used
        self.resampling_stride = self.__format_stride__(resampling_stride)
        # Calculate the channel multipliers per downsampling operation
        self.channel_multipliers = [
            int(np.prod(stride)) for stride in self.resampling_stride
        ]
        self.resampling_method = resampling_method
        self.resampling_init = resampling_init
        if resampling_kwargs is None:
            resampling_kwargs = {}
        self.resampling_kwargs = resampling_kwargs
        # Calculate the total downsampling factor per spatial dimension
        self.downsampling_factors = self.__total_downsampling_factor__(
            self.resampling_stride
        )

        # --- Check architecture and print ---
        self.check_architecture(channels, architecture)
        self.print_layout()

        # --- Verbosity level ---
        self.verbose = verbose

        # --- Create the architecture of the iUNet ---
        downsampling_op = InvertibleDownsampling1D
        upsampling_op = InvertibleUpsampling1D

        self.input_nodes = list()
        self.encoder_modules = list()
        self.decoder_modules = list()
        self.slice_layers = list()
        self.concat_layers = list()
        self.downsampling_layers = list()
        self.upsampling_layers = list()
        self.output_nodes = list()

        self.min_input_size = 12 * subdivision ** 2  # resampling_stride ** (len(channels) - 1)
        base_shape = (self.in_channels,) + (self.min_input_size,) * self.dim
        current_node = Ff.InputNode(*base_shape, name='input')
        self.input_nodes.append(current_node)

        sampling_arguments = {"method": self.resampling_method,
                              "stride": resampling_stride,
                              "init": self.resampling_init,
                              "learnable": learnable_resampling,
                              **self.resampling_kwargs,
                              }

        convolution_args = dict(dim=self.dim,
                                normalization=normalization_conv,
                                padding_mode=padding_mode,
                                inv_block_kwargs=inv_block_kwargs,
                                coupling_layer=coupling_layer)

        # --- Create the encoder of the iUNet ---
        for num, (chs, arch, n_conv) in enumerate(zip(self.channels, self.architecture, self.conv_depth)):
            # Define Convolutional block
            current_node = Ff.Node(current_node.out0,
                                   IUnetCOnvBlock,
                                   {'laplacian': self.laplacians[num], 'depth': arch, 'n_conv': n_conv,
                                    **convolution_args},
                                   name=f'encoder_conv_lev_{num}_{chs}')
            self.encoder_modules.append(current_node)

            # Add downsampling element
            if num != len(self.channels) - 1:
                down_channels = self.channels_before_downsampling[num]
                current_node = Ff.Node(current_node.out0, Fm.Split,
                                       {'section_sizes': (chs - down_channels, down_channels), 'dim': 0},
                                       name=f'split_lev_{num}')
                self.slice_layers.append(current_node)

                current_node = Ff.Node(current_node.out1,
                                       downsampling_op,
                                       sampling_arguments,
                                       name=f'downsample_lev_{num}')

                self.downsampling_layers.append(current_node)

        # --- Create the decoder of the iUNet ---
        for num, (chs, arch, n_conv) in reversed(
            list(enumerate(zip(self.channels, self.architecture, self.conv_depth)))):

            # Concatenate with skip connections(except bottleneck)
            if num != len(self.channels) - 1:
                skip_connection = self.slice_layers[num]
                current_node = Ff.Node([current_node.out0, skip_connection.out0], Fm.Concat,
                                       {'dim': 0}, name=f'concat_lev_{num}')
                self.concat_layers.append(current_node)

            # Define Convolutional block
            current_node = Ff.Node(current_node.out0,
                                   IUnetCOnvBlock,
                                   {'laplacian': self.laplacians[num], 'depth': arch, 'n_conv': n_conv,
                                    **convolution_args},
                                   name=f'decoder_conv_lev_{num}_{chs}')
            self.decoder_modules.append(current_node)

            # Add upsampling element
            if num != 0:
                current_node = Ff.Node(current_node.out0,
                                       upsampling_op,
                                       sampling_arguments,
                                       name=f'upsample_lev_{num}')

                self.upsampling_layers.append(current_node)

        self.output_nodes.append(Ff.OutputNode(current_node, name='output'))
        # Initialize the GraphINN from FreIA
        element_list = [*self.input_nodes,
                        *self.encoder_modules,
                        *self.decoder_modules,
                        *self.downsampling_layers,
                        *self.upsampling_layers,
                        *self.slice_layers,
                        *self.concat_layers,
                        *self.output_nodes]

        super().__init__(element_list)

    @staticmethod
    def check_channels(channels):
        channels_out = [channels[0]]
        for i in range(1, len(channels)):
            current_ch = channels[i]
            out_ch = int(max(1, np.round(current_ch / 4)) * 4)
            channels_out.append(out_ch)

        if not all([in_ch == out_ch for in_ch, out_ch in zip(channels, channels_out)]):
            print("The number of channels in the downsampled channels must be multiple of 4")
            print(f"Desired number of channels changed from {channels}, to {channels_out}")

        return channels_out

    def check_architecture(self, channels, architecture):
        # --- Setting up the required channel numbers ---
        # The user-specified channel numbers can potentially not be enforced.
        # Hence, we choose the best possible approximation.
        desired_channels = channels
        channel_errors = []  # Measure how far we are off.

        for i in range(len(architecture) - 1):
            factor = desired_channels[i + 1] / self.channels[i]
            skip_fraction = (self.channel_multipliers[i] - factor) / self.channel_multipliers[i]
            self.skipped_channels.append(int(max([1, np.round(self.channels[i] * skip_fraction)])))
            self.channels_before_downsampling.append(self.channels[i] - self.skipped_channels[-1])
            self.channels.append(self.channel_multipliers[i] * self.channels_before_downsampling[i])
            channel_errors.append(abs(self.channels[i] - desired_channels[i]) / desired_channels[i])

        if list(channels) != list(self.channels):
            print(
                f"""
                Could not exactly create an iUNet with channels={channels} and "
                resampling_stride={self.resampling_stride}. Instead using closest achievable "
                configuration: channels={self.channels}. Average relative error: {np.mean(channel_errors)}
                """
            )

    def __format_stride__(self, stride):
        """Parses the resampling_stride and reformats it into a standard format.
        """
        self.__check_stride_format__(stride)
        if isinstance(stride, int):
            return [(stride,) * self.dim] * (self.num_levels - 1)
        if isinstance(stride, tuple):
            return [stride] * (self.num_levels - 1)
        if isinstance(stride, list):
            for i, element in enumerate(stride):
                if isinstance(element, int):
                    stride[i] = (element,) * self.dim
            return stride

    def __check_stride_format__(self, stride):
        """Check whether the stride has the correct format to be parsed.
        The format can be either a single integer, a single tuple (where the
        length corresponds to the spatial dimensions of the data), or a list
        containing either of the last two options (where the length of the
        list has to be equal to the number of downsampling operations),
        e.g. ``2`, ``(2,1,3)``, ``[(2,1,3), (2,2,2), (4,3,1)]``.
        """

        def raise_format_error():
            raise AttributeError(
                "resampling_stride has the wrong format. "
                "The format can be either a single integer, a single tuple "
                "(where the length corresponds to the spatial dimensions of the "
                "data), or a list containing either of the last two options "
                "(where the length of the list has to be equal to the number "
                "of downsampling operations), e.g. 2, (2,1,3), "
                "[(2,1,3), (2,2,2), (4,3,1)]. "
            )

        if isinstance(stride, int):
            pass
        elif isinstance(stride, tuple):
            if len(stride) == self.dim:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_format_error()
        elif isinstance(stride, list):
            if len(stride) == self.num_levels - 1:
                for element in stride:
                    self.__check_stride_format__(element)
            else:
                raise_format_error()
        else:
            raise_format_error()

    def __total_downsampling_factor__(self, stride):
        """Calculates the total downsampling factor per spatial dimension.
        """
        factors = [1] * len(stride[0])
        for i, element_tuple in enumerate(stride):
            for j, element_int in enumerate(stride[i]):
                factors[j] = factors[j] * element_int
        return tuple(factors)

    def print_layout(self):
        """Prints the layout of the iUNet.
        """
        print_iunet_layout(self)
