"""Insert docs here...

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import typing

# Third party imports
import torch
import torchvision
from torch.hub import load_state_dict_from_url


# ----------------------------------------------------------------------
# class
# ----------------------------------------------------------------------

SIZE_TO_CFG_MAP = {
    11: torchvision.models.vgg.cfgs['A'],
    13: torchvision.models.vgg.cfgs['B'],
    16: torchvision.models.vgg.cfgs['D'],
    19: torchvision.models.vgg.cfgs['E']
}


# ----------------------------------------------------------------------
# class
# ----------------------------------------------------------------------

class VGG(torchvision.models.VGG):
    """Extending the torchvision model to add visibility.

    """
    def __init__(self, features: torch.nn.Sequential, num_classes: int = 1000,
                 init_weights: bool = True):
        super().__init__(features, num_classes, init_weights)

    @typing.overload
    def forward(self, x: torch.Tensor,
                watch_layers: bool = False) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor,
                watch_layers: typing.Optional[bool] = False) \
            -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]:
        """Save the output of ReLU layers to cache and forward propagate

        For each ReLU layer that the input tensor is forwarded for, save
        the results. This is done to provide visibility into the model
        when computing loss functions to perform style transfers. Only,
        the output of ReLU layers are saved because ReLU layers follow
        Conv2D layers in VGG, and we are interested in the activated
        feature maps when calculating the style transfer losses.

        Notes
        -----
        The naming convention for the layers follows the convention of
        the original publication:

        `"Very Deep Convolutional Networks For Large-Scale Image
        Recognition"
        <https://arxiv.org/pdf/1409.1556.pdf>`_

        The nameing convention is "conv{group_no}_{layer_no}

        Parameters
        ----------
        x
            Input image to classify.
        watch_layers
            If True, also return a dictionary of outputs from the
            activated feature maps (the default is False)

        Returns
        -------
        Tensor containing output results and layer outputs if enabled.

        """
        group_no, layer_no = 1, 1
        layer_output_map: typing.Dict[str, torch.Tensor] = {}
        for module in self.features.children():
            x = module(x)
            if isinstance(module, torch.nn.ReLU):
                layer_output_map[f"conv{group_no}_{layer_no}"] = x
                layer_no += 1
            elif isinstance(module, torch.nn.MaxPool2d):
                group_no += 1
                layer_no = 1

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if watch_layers:
            return x, layer_output_map

        return x


def vgg(size: int, batch_norm: typing.Optional[bool] = False,
        pretrained: typing.Optional[bool] = True,
        **kwargs):
    """Construct the model based on the  given config.

    Based on the function to create models from torchvision.

    Parameters
    ----------
    size
        Size of network, options are 11, 13, 16, and 19.
    batch_norm
        Whether batch normalization layers should be used.
    pretrained
        If True, pretrained weights are loaded into the model

    Returns
    -------
    VGG model based on provided architecture.

    """
    if size not in {11, 13, 16, 19}:
        raise TypeError("value for size is invalid")

    arch_str = f"vgg{size}" + "_bn" * batch_norm

    if pretrained:
        kwargs["init_weights"] = False
    feats = torchvision.models.vgg.make_layers(SIZE_TO_CFG_MAP[size],
                                               batch_norm=batch_norm)
    model = VGG(feats, **kwargs)

    if pretrained:
        model_url = torchvision.models.vgg.model_urls[arch_str]
        state_dict = load_state_dict_from_url(model_url)
        model.load_state_dict(state_dict)

    return model
