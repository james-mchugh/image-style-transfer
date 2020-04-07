"""Loss Functions for performing style transfers.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# Third party imports
import torch
from torch.nn import functional

# Local application imports


# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# class
# ----------------------------------------------------------------------

class ContentLoss(torch.nn.Module):
    """Computes loss between the content and generated representations

    """

    def __init__(self, content: torch.Tensor):
        super().__init__()
        self.content = content
        self.loss: torch.Tensor = torch.Tensor()

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the generated image and content.
        
        Parameters/
        ----------
        generated
            Generated image representation.

        Returns
        -------
        Tensor containing a rendition of the L2Loss between the tensors.

        """
        self.loss = functional.mse_loss(self.content, generated)
        return generated


class StyleLoss(torch.nn.Module):
    """Computes loss between the style and generated representations

    """

    def __init__(self, style: torch.Tensor):
        super().__init__()
        self.style_gram = self.gram_matrix(style)
        self.loss: torch.Tensor = torch.Tensor()

    def forward(self, generated: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the generated image and style.

        Parameters
        ----------
        generated
            Generated image representation.

        Returns
        -------
        Tensor containing a rendition of the L2Loss between the style
        and content gram matrix.

        """
        generated_gram = self.gram_matrix(generated)
        self.loss = functional.mse_loss(generated_gram, self.style_gram)
        return generated

    @staticmethod
    def gram_matrix(x: torch.Tensor) -> torch.Tensor:
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
        x
            A 3D volume in which the first layer is the number of
            channels in the tensor.

        Returns
        -------
        A tensor containing the gram matrix between the two tensors.

        """
        b, d, h, w = x.shape
        features = x.view(b * d, h * w)
        gram = torch.mm(features, features.t())

        return gram / (b * d * h * w)
