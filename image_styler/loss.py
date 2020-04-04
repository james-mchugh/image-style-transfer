"""Loss Functions for performing style transfers.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import operator
import typing

# Third party imports
import torch

# Local application imports


# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# class
# ----------------------------------------------------------------------

class ContentLoss(torch.nn.Module):

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, content: torch.Tensor, generated: torch.Tensor,
                layer_weight: typing.Optional[float] = 1.0) -> torch.Tensor:
        """Compute the loss between the generated image and content.
        
        Parameters/
        ----------
        content
            Content representation to match.
        generated
            Generated image representation.
        layer_weight
            If the layer has special weight, that can be applied here.
            All layer weights should add to 1. (default is 1).

        Returns
        -------
        Tensor containing a rendition of the L2Loss between the tensors.

        """
        loss = 1/2 * (generated - content).pow(2).sum()
        loss *= self.weight * layer_weight
        return loss


class StyleLoss(torch.nn.Module):

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def forward(self, style: torch.Tensor, generated: torch.Tensor,
                layer_weight: typing.Optional[float] = 1.0) -> torch.Tensor:
        """Compute the loss between the generated image and style.

        Parameters
        ----------
        style
            Style representation to match.
        generated
            Generated image representation.
        layer_weight
            If the layer has special weight, that can be applied here.
            All layer weights should add to 1. (default is 1).

        Returns
        -------
        Tensor containing a rendition of the L2Loss between the style
        and content gram matrix.

        """
        generated_gram = self.gram(generated)
        nl, ml = generated.shape[1], operator.mul(*generated.shape[2:])
        loss = 1 / (4*(nl*ml)**2) * (generated_gram - style).pow(2).sum()
        loss *= self.weight * layer_weight
        return loss

    @staticmethod
    def gram(feat_map: torch.Tensor) -> torch.Tensor:
        """Calculate the gram matrix between of the Volume.

        The gram matrix is computed by flattening the dimensions
        corresponding to the height and weight of the feature map.
        After this, the dot product is taken between the feature map
        vectors, and the result is stored in the gram matrix. The
        element G[i, j] of the gram matrix represents the dot between
        layers i and j.

        The Tensor is flattened beyond the first dimension prior to
        computing the gram matrix. This is to accommodate for 3
        dimensional tensors in which the first layer represents the
        number of channels in the volume.

        The gram (or Gramian matrix) is the inner product between the
        vectors. More details can be found here
        https://en.wikipedia.org/wiki/Gramian_matrix.

        Parameters
        ----------
        feat_map
            A 3D volume in which the first layer is the number of
            channels in the tensor.

        Returns
        -------
        A tensor containing the gram matrix between the two tensors.

        """
        feat_map = feat_map.flatten(1)

        return feat_map.matmul(feat_map.transpose(0, 1))
