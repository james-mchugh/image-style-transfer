"""Insert docs here...

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import enum


# ----------------------------------------------------------------------
# enums
# ----------------------------------------------------------------------

class InitMethod(str, enum.Enum):
    RANDOM = "random"
    CONTENT = "content"
    STYLE = "style"


class PoolType(str, enum.Enum):
    MAX = "max"
    AVG = "avg"


# ----------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------

DEFAULT_CONTENT_WEIGHT = 1
DEFAULT_STYLE_WEIGHT = 1e6

DEFAULT_CONTENT_LAYERS = ["conv4_4"]
DEFAULT_STYLE_LAYERS = [
    "conv1_1",
    "conv2_1",
    "conv3_1",
    "conv4_1",
    "conv5_1"
]

DEFAULT_INIT_METHOD = InitMethod.RANDOM
DEFAULT_POOL_TYPE = PoolType.AVG

DEFAULT_MAX_IMAGE_SIZE = 512
DEFAULT_NUM_ITER = 2000
DEFAULT_LEARNING_RATE = 1.0
