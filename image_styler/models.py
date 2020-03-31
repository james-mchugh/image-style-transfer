"""Models for the image styler.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports

# Third party imports
import torch

# Local application imports
from . import utils

# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------
import typing

CONV_STRIDE = (1, 1)
CONV_KERNEL_SIZE = (3, 3)
CONV_PADDING = (1, 1)

POOL_STRIDE = (2, 2)
POOL_KERNEL_SIZE = (2, 2)


# ----------------------------------------------------------------------
# class
# ----------------------------------------------------------------------

class VGG19(torch.nn.Module):
    """Class for the VGG19 model for image classification.

    """

    def __init__(self):
        super().__init__()
        self.conv_01 = self.ConvReLU(3, 64)
        self.conv_02 = self.ConvReLU(64, 64)
        self.pool_01 = torch.nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE)

        self.conv_03 = self.ConvReLU(64, 128)
        self.conv_04 = self.ConvReLU(128, 128)
        self.pool_02 = torch.nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE)

        self.conv_05 = self.ConvReLU(128, 256)
        self.conv_06 = self.ConvReLU(256, 256)
        self.conv_07 = self.ConvReLU(256, 256)
        self.conv_08 = self.ConvReLU(256, 256)
        self.pool_03 = torch.nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE)

        self.conv_09 = self.ConvReLU(256, 512)
        self.conv_10 = self.ConvReLU(512, 512)
        self.conv_11 = self.ConvReLU(512, 512)
        self.conv_12 = self.ConvReLU(512, 512)
        self.pool_04 = torch.nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE)

        self.conv_13 = self.ConvReLU(512, 512)
        self.conv_14 = self.ConvReLU(512, 512)
        self.conv_15 = self.ConvReLU(512, 512)
        self.conv_16 = self.ConvReLU(512, 512)
        self.pool_05 = torch.nn.MaxPool2d(POOL_KERNEL_SIZE, POOL_STRIDE)

        # noinspection PyUnresolvedReferences
        self.flatten = torch.nn.Flatten()
        self.dense_01 = torch.nn.Linear(512 * 7 * 7, 4096)
        self.act_01 = torch.nn.ReLU()
        self.dense_02 = torch.nn.Linear(4096, 4096)
        self.act_02 = torch.nn.ReLU()
        self.dense_03 = torch.nn.Linear(4096, 1000)
        self.act_03 = torch.nn.Softmax(dim=1000)

    def forward(self, x) -> torch.Tensor:
        out = torch.Tensor()
        for module in self.children():
            out = module(out if len(out) > 0 else x)

        return out

    @property
    def weighted_modules(self) -> typing.Iterator[torch.nn.Module]:
        """Get the weighted modules of the class.

        Returns
        -------
        An iterator of weighted modules.

        """
        return filter(lambda mod: hasattr(mod, "weight"), self.modules())

    def set_layer_parameters(self,
                             parameters: typing.Iterator[utils.Parameters]):
        """Set the weights of the network.

        Parameters
        ----------
        parameters
            Weights to set for the layers

        """
        for module, parameters in zip(self.weighted_modules, parameters):
            module.register_parameter('weight',
                                      torch.nn.Parameter(
                                          parameters.weight.reshape(
                                              module.weight.shape), True))
            module.register_parameter('bias',
                                      torch.nn.Parameter(
                                          parameters.bias.reshape(
                                              module.bias.shape), False))

    class ConvReLU(torch.nn.Module):
        """Convolution layer with ReLU.
        
        The stride, padding, and kernel are static as defined in VGG-19
        
        """

        def __init__(self, in_channels: int, out_channels: int,
                     weights: typing.Optional[torch.nn.Parameter] = None):
            """Create a convolution layer with ReLU activation

            Parameters
            ----------
            in_channels
                Number of channels coming into the layer.
            out_channels
                Number of channels coming out of the layer.
            weights
                Weights to initialize the layer with.
    
            Returns
            -------
            Convolution Module with proper channels and ReLU activation.
    
            """
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                        CONV_KERNEL_SIZE, CONV_STRIDE,
                                        CONV_PADDING)
            self.activation = torch.nn.ReLU()

            if weights:
                self.conv.register_parameter("weight", weights)

        def forward(self, x):
            return self.activation(self.conv(x))
