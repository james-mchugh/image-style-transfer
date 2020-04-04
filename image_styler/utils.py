"""Insert docs here...

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import typing

# Third party imports
import cv2
import torch
import numpy as np


# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------

class Parameters(typing.NamedTuple):
    bias: torch.nn.Parameter
    weight: torch.nn.Parameter


# ----------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------


def load_image(file_name: str, max_size: typing.Optional[int] = 0) \
        -> np.ndarray:
    """Load the image and prepare it to be loaded into the network.

    Parameters
    ----------
    file_name
        Path to the image to be loaded.
    max_size
        Max size for the height or width. If nonzero, the image will be
        reshaped accordingly.

    Returns
    -------
    Array containing data of image.

    """
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"no image found at {file_name}")

    img = img.astype(np.float64)
    if max_size > 0:
        img = reshape_image(img, max_size)

    img = preprocess_image(img)

    return img


def reshape_image(img: np.ndarray, max_size: int) -> np.ndarray:
    """Reshape the image so that its height and width are less than max.

    Parameters
    ----------
    img
        Image to be reshaped.
    max_size
        Max size of the height and width of the image.

    Returns
    -------
    Reshaped image so that height and width are below max_size.

    """
    height, width, _ = img.shape
    ratio = height / width

    long_side = max(height, width)
    needs_reshape = long_side > max_size

    if needs_reshape:
        new_height = int(max_size if height == long_side else max_size * ratio)
        new_width = int(max_size if width == long_side else max_size / ratio)
        img = cv2.resize(img, dsize=(new_width, new_height),
                         interpolation=cv2.INTER_AREA)

    return img


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess the image so it is ready to be used in a model.

    Convert BGR images from OpenCV to RGB and adds an extra dimension.

    Parameters
    ----------
    img
        BGR image to be preprocessed.

    Returns
    -------
    Processed image.

    """
    img = np.flip(img, axis=2).copy()  # BGR to RGB
    img = img.reshape(1, *img.shape[::-1])  # Add extra dimension

    return img


def save_image(img: np.ndarray, file: str):
    """Save the tensor as an image.

    Parameters
    ----------
    img
        Array to save as image.
    file
        File to save tensor to.

    """
    img = img.reshape(img.shape[-1:0:-1])
    img = np.flip(img, axis=2).copy()
    cv2.imwrite(file, img)
