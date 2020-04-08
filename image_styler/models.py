"""PyTorch models used to perform style transfer

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import typing

# Third party imports
import torch
import torchvision

# Local application imports
from . import loss, constants

# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------

SIZE_TO_CFG_MAP = {
    11: torchvision.models.vgg.cfgs['A'],
    13: torchvision.models.vgg.cfgs['B'],
    16: torchvision.models.vgg.cfgs['D'],
    19: torchvision.models.vgg.cfgs['E']
}

CNN_NORMALIZATION_MEAN = torch.tensor([0.485, 0.456, 0.406])
CNN_NORMALIZATION_STD = torch.tensor([0.229, 0.224, 0.225])


# ----------------------------------------------------------------------
# class
# ----------------------------------------------------------------------

class StyleVGG(torch.nn.Module):
    """Extending the torchvision VGG model to add visibility.

    """

    def __init__(self, vgg: torch.nn.Module, content: torch.Tensor, style: torch.Tensor,
                 content_layers: typing.List[str], style_layers: typing.List[str],
                 device: torch.device,
                 pool_type: constants.PoolType = constants.DEFAULT_POOL_TYPE):
        """Initialze the StyleVGG model and build it.

        Parameters
        ----------
        vgg
            Base model to construct style transfer model. In theory,
            this could be any CNN, but it is currently configured to
            expect torchvision VGG models with a Sequential Module
            stored in vgg.features
        content
            Content image to create representations of for loss
            functions
        style
            Style image to create representations of for loss functions
        content_layers
            Layers to use for content representations when computing
            content loss.
        style_layers
            Layers to use for style representations when computing
            style loss.
        device
            Device to use for computation.
        pool_type
            Type of pooling layers, either max or average.
        """
        super().__init__()
        normalizer = Normalizer(CNN_NORMALIZATION_MEAN.to(device),
                                CNN_NORMALIZATION_STD.to(device))
        self.features = torch.nn.Sequential(normalizer)
        self.content_layers = set(content_layers)
        self.style_layers = set(style_layers)
        self.content_losses = []
        self.style_losses = []
        self.build(vgg, content, style, pool_type)
        self.to(device)
        self.eval()

    def build(self, vgg: torchvision.models.VGG,
              content: torch.Tensor, style: torch.Tensor,
              pool_type: constants.PoolType):
        """Build a model with feature generation from the provided model

        Given a VGG model from torchvision, rebuild a model with the
        original model's features while adding in layers to calculate
        the content and style losses. The pool type can also be swapped
        out. In the traditional VGG model, max pooling is used, but here
        average pooling is used as it appears to give better results.

        Parameters
        ----------
        vgg
            Base model to construct style transfer model. In theory,
            this could be any CNN, but it is currently configured to
            expect torchvision VGG models with a Sequential Module
            stored in vgg.features
        content
            Content image to create representations of for loss
            functions
        style
            Style image to create representations of for loss functions
        pool_type
            Type of pooling layers, either max or average.

        Notes
        -----
        The naming convention for the layers follows the convention of
        the original publication:

        `"Very Deep Convolutional Networks For Large-Scale Image
        Recognition"
        <https://arxiv.org/pdf/1409.1556.pdf>`_

        The naming convention is "conv{group_no}_{layer_no}

        """
        group_num, relu_num, conv_num = 1, 1, 1
        for layer in vgg.features.children():
            if isinstance(layer, torch.nn.Conv2d):
                name = f"conv{group_num}_{conv_num}"
                conv_num += 1

            elif isinstance(layer, torch.nn.ReLU):
                name = f"relu{group_num}_{relu_num}"
                layer = torch.nn.ReLU(inplace=False)
                relu_num += 1

            elif isinstance(layer, torch.nn.MaxPool2d):
                name = f"max{group_num}"
                if pool_type == constants.PoolType.AVG:
                    layer = torch.nn.AvgPool2d(layer.kernel_size, layer.stride, layer.padding)
                else:
                    layer = torch.nn.MaxPool2d(layer.kernel_size, layer.stride, layer.padding)
                group_num += 1
                conv_num, relu_num = 1, 1

            else:
                raise TypeError("Invalid layer type received.")

            self.features.add_module(name, layer)

            if name in self.content_layers:
                name = f"content_loss{len(self.content_losses)}"
                content_repr = self.features(content).detach()
                content_loss = loss.ContentLoss(content_repr)
                self.content_losses.append(content_loss)
                self.features.add_module(name, content_loss)

            if name in self.style_layers:
                name = f"style_loss{len(self.style_losses)}"
                style_repr = self.features(style).detach()
                style_loss = loss.StyleLoss(style_repr)
                self.style_losses.append(style_loss)
                self.features.add_module(name, style_loss)

            if (len(self.style_layers) == len(self.style_losses)
                    and len(self.content_layers) == len(self.content_losses)):
                # we have all the layers we need, let's wrap it up
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the input through and generate its representation.

        Parameters
        ----------
        x
            Input image to generate representations of.

        Returns
        -------
        Tensor containing representation of input.

        """
        x = self.features.forward(x)

        return x


class Normalizer(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super(Normalizer, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = (x - self.mean) / self.std
        return norm
