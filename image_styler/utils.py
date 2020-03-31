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
import scipy.io


# Local application imports


# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------

class Parameters(typing.NamedTuple):
    bias: torch.nn.Parameter
    weight: torch.nn.Parameter


# ----------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------


def load_mat_model(file_name: str) -> typing.List[Parameters]:
    """Load a MatLab model of containing network weights.

    The format for the model is presumed to be the same format used in
    the model defined here:

    https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat

    Parameters
    ----------
    file_name
        Path to file to open

    Returns
    -------
    List of Parameters tuples containing biases and weights.

    """
    parameters_list = []
    mat = scipy.io.loadmat(file_name)
    for layer in mat['layers'][0]:

        try:
            has_weights = isinstance(layer[0][0][2][0][0], np.ndarray)
        except IndexError:
            has_weights = False

        if not has_weights:
            continue

        weight_array: np.ndarray = layer[0][0][2][0][0]
        weight_array = weight_array.reshape(weight_array.shape[-1::-1])
        weight = torch.nn.Parameter(torch.from_numpy(weight_array),
                                    requires_grad=True)

        bias_array = layer[0][0][2][0][1].flatten()
        bias = torch.nn.Parameter(torch.from_numpy(bias_array),
                                  requires_grad=True)

        parameters_list.append(Parameters(bias, weight))

    return parameters_list


def load_image(file_name: str, max_size: typing.Optional[int] = 0) \
        -> torch.Tensor:
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
    Tensor containing data of image.

    """
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = img.astype(np.float64)

    if max_size > 0:
        img = reshape_image(img, max_size)

    img = preprocess_image(img)

    return torch.from_numpy(img)


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

    base = max(height, width)
    needs_reshape = base > max_size

    if needs_reshape:
        new_height = int(max_size if height == base else max_size * ratio)
        new_width = int(max_size if width == base else max_size / ratio)
        img = cv2.resize(img, dsize=(new_height, new_width),
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
