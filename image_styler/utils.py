"""Useful utils for interacting with images.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import typing

# Third party imports
import torch
import torchvision
from PIL import Image


# ----------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------

def load_image(file_name: str, max_size: typing.Optional[int] = 0,
               new_shape: typing.Optional[typing.Tuple[int, int]] = None) -> torch.Tensor:
    """Load the image and prepare it to be loaded into the network.

    Parameters
    ----------
    file_name
        Path to the image to be loaded.
    max_size
        Max size for the height or width. If nonzero, the image will be
        reshaped accordingly.
    new_shape
        If passed, resize the image to this exact shape.

    Returns
    -------
    Tensor containing image data.

    """
    img = Image.open(file_name)
    if new_shape is None:
        if max_size > 0:
            new_shape = _get_new_shape(img, max_size)
        else:
            new_shape = img.shape

    loader = torchvision.transforms.Compose([
        torchvision.transforms.Resize(new_shape),
        torchvision.transforms.ToTensor()
    ])

    img = loader(img).unsqueeze(0)

    return img


def _get_new_shape(img: Image, max_size: int) -> typing.Tuple[int, int]:
    """Get new shape for image based on the max size of either side.

    Parameters
    ----------
    img
        Array containing image data to reshape.
    max_size
       The max size of either of the sides of the image.

    Returns
    -------
    Tuple containing the new height and width of the image.

    """
    height, width = img.height, img.width
    ratio = height / width

    long_side = max(height, width)

    new_height = int(max_size if height == long_side else max_size * ratio)
    new_width = int(max_size if width == long_side else max_size / ratio)

    return new_height, new_width


def save_image(img: torch.Tensor, file: str):
    """Save the tensor as an image.

    Parameters
    ----------
    img
        Tensor to save as image.
    file
        File to save tensor to.

    """
    unloader = torchvision.transforms.ToPILImage()
    pil_img: Image.Image = unloader(img.squeeze(0))
    pil_img.save(file)
