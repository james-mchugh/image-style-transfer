"""Where the styling happens.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import typing
import functools

# Third party imports
import torch
from torch.optim.lbfgs import LBFGS
from torch.optim.optimizer import Optimizer
import torchvision

# Local application imports
from . import constants, models


# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------

class Stylizer(object):
    """This class facilitates the style transfers for images.

    Given two images, a content image and a style image, this class
    facilitates the transfer of the style from the style image to the
    content image. It does this by generating a tensor with the same
    shape as the content image, but with randomly initialized values.
    It is then optimized to reduce the distance of its representations
    from the representations of the style and content images.

    References
    ----------
    [1] L. A. Gatys, A. S. Ecker and M. Bethge, "Image Style Transfer
    Using Convolutional Neural Networks," 2016 IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016,
    pp. 2414-2423.

    """

    def __init__(self, content: torch.Tensor, style: torch.Tensor,
                 content_layers: typing.List[str] = constants.DEFAULT_CONTENT_LAYERS,
                 style_layers: typing.List[str] = constants.DEFAULT_STYLE_LAYERS,
                 pool_type: constants.PoolType = constants.DEFAULT_POOL_TYPE,
                 device: torch.device = torch.device("cpu")):
        self.content = content.to(device)
        self.style = style.to(device)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.pool_type = pool_type
        self.device = device
        self.iter_num = 0

        vgg = torchvision.models.vgg19(True).to(device).eval()
        self.model = models.StyleVGG(vgg, self.content, self.style, content_layers,
                                     style_layers, self.device)

    def stylize(self, init_method: constants.InitMethod = constants.DEFAULT_INIT_METHOD,
                content_weight: float = constants.DEFAULT_CONTENT_WEIGHT,
                style_weight: float = constants.DEFAULT_STYLE_WEIGHT,
                content_layer_weights: typing.List[float] = None,
                style_layer_weights: typing.List[float] = None,
                max_iter: int = constants.DEFAULT_NUM_ITER,
                learning_rate: float = constants.DEFAULT_LEARNING_RATE) -> torch.Tensor:
        """Stylize the image.

        Parameters
        ----------
        init_method
            Method to initialize the generated image.
        content_weight
            Weight for the content portion of the loss.
        style_weight
            Weight for the style portion of the loss.
        content_layer_weights
            Weights for each of the content layers being used.
        style_layer_weights
            Weights for each of the style layers being used
        max_iter
            Maximum number of iterations for the optimization.
        learning_rate
            Rate at which to traverse the loss function.

        Returns
        -------
        A tensor containing the generated image.

        """
        generated = self.init_image(init_method)

        if not content_layer_weights:
            content_layer_weights = self.init_layer_weights(self.content_layers)

        if not style_layer_weights:
            style_layer_weights = self.init_layer_weights(self.style_layers)

        if len(content_layer_weights) != len(self.content_layers):
            raise ValueError("Invalid number of content weights.")

        if len(style_layer_weights) != len(self.style_layers):
            raise ValueError("Invalid number of style weights.")

        optimizer = LBFGS((generated,), max_iter=max_iter, lr=learning_rate)

        closure = functools.partial(self.optim_iteration, generated,optimizer,
                                    content_weight, style_weight,
                                    content_layer_weights, style_layer_weights)
        optimizer.step(closure)

        return generated.clamp(0, 1).detach().cpu()

    @staticmethod
    def init_layer_weights(layers: typing.List[str]) -> typing.List[float]:
        """Initialize the weights of individual layers of the model.

        Parameters
        ----------
        layers
            Names of layers to initialize weights for.

        Returns
        -------
        Uniform weights corresponding to the layers.

        """
        num_layers = len(layers)
        return [1/num_layers] * num_layers

    def init_image(self, init_method: constants.InitMethod) -> torch.Tensor:
        """Initialize image to be generated.

        Parameters
        ----------
        init_method
            Method for initializing the base image.

        Returns
        -------
        Tensor to be optimized to match style and content.

        """
        if init_method is constants.InitMethod.RANDOM:
            out = torch.randn(self.content.shape, device=self.device)
        elif init_method is constants.InitMethod.CONTENT:
            out = self.content.clone()
        else:
            out = self.style.clone()

        return out.requires_grad_()

    def optim_iteration(self, generated: torch.Tensor,
                        optimizer: Optimizer,
                        content_weight: float, style_weight: float,
                        content_layer_weights: typing.List[float],
                        style_layer_weights: typing.List[float]):
        """Run an iteration of the optimization process.

        Parameters
        ----------
        generated
            Generated image to optimize.
        optimizer
            Optimizer to use.
        content_weight
            Weight for the content portion of the loss.
        style_weight
            Weight for the style portion of the loss.
        content_layer_weights
            Weights for each of the content layers being used.
        style_layer_weights
            Weights for each of the style layers being used

        Returns
        -------
        Tensor containing the overall style transfer loss.

        """
        optimizer.zero_grad()
        self.model(generated)

        content_loss = torch.tensor(0.0, device=self.device)
        style_loss = torch.tensor(0.0, device=self.device)
        for layer_loss, weight in zip(self.model.content_losses, content_layer_weights):
            content_loss += weight * layer_loss.loss
        content_loss *= content_weight

        for layer_loss, weight in zip(self.model.style_losses, style_layer_weights):
            style_loss += weight * layer_loss.loss
        style_loss *= style_weight

        transfer_loss = content_loss + style_loss
        transfer_loss.backward()

        if self.iter_num % 50 == 0:
            print(f"Iter {self.iter_num}\n"
                  f"\tContent Loss: {float(content_loss): 0.04f}\n"
                  f"\tStyle Loss: {float(style_loss): 0.04f}\n"
                  f"\tTransfer Loss: {float(transfer_loss): 0.04f}\n")
        self.iter_num += 1

        return transfer_loss
