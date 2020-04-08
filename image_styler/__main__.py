"""Generate a stylized image based on the provided images.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import argparse

# Third party imports
import torch

# Local application imports
from . import utils, stylize, constants


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("content_img",
                        help="Path to the content image to stylize")
    parser.add_argument("style_img",
                        help="Path to the style image to transfer style from")

    parser.add_argument("output_file",
                        help="Path to output file")

    parser.add_argument("--max-img-size", "-s", type=int,
                        default=constants.DEFAULT_MAX_IMAGE_SIZE,
                        help="The max size for both the height and width of "
                             "the image")

    parser.add_argument("--init-method",
                        type=constants.InitMethod,
                        default=constants.DEFAULT_INIT_METHOD,
                        choices=[e.value for e in constants.InitMethod],
                        help="Strategy for generating the init image.")

    parser.add_argument("--gpu",
                        help="Use the GPU (if available)",
                        action="store_true")

    parser.add_argument("--pool-type",
                        type=constants.PoolType,
                        default=constants.DEFAULT_POOL_TYPE,
                        choices=[e.value for e in constants.PoolType],
                        help="Type of pooling to use in VGG.")

    parser.add_argument("--content-weight", type=float,
                        default=constants.DEFAULT_CONTENT_WEIGHT,
                        help="Weighting for the content portion of the loss function")

    parser.add_argument("--style-weight", type=float,
                        default=constants.DEFAULT_STYLE_WEIGHT,
                        help="Weighting for the style portion of the loss function")

    parser.add_argument("--content-layers", nargs="+",
                        default=constants.DEFAULT_CONTENT_LAYERS,
                        help="Layers to use to extract content representation")

    parser.add_argument("--style-layers", nargs="+",
                        default=constants.DEFAULT_STYLE_LAYERS,
                        help="Layers to use to extract style representation")

    parser.add_argument("--content-layer-weights", nargs='+',
                        default=[], type=float,
                        help="Weights for each layer used to create content "
                             "representation. This should be the same length "
                             "as the number of content layers used. By "
                             "default, equal weighting is used.")

    parser.add_argument("--style-layer-weights", nargs='+',
                        default=[], type=float,
                        help="Weights for each layer used to create style "
                             "representation. This should be the same length "
                             "as the number of style layers used. By "
                             "default, equal weighting is used.")

    parser.add_argument("--max-iter", "-i", type=int,
                        default=constants.DEFAULT_NUM_ITER,
                        help="Number of LBFGS optimization iterations to use.")

    parser.add_argument("--learning-rate", "-l", type=float,
                        default=constants.DEFAULT_LEARNING_RATE,
                        help="The rate at which the optimizer should traverse "
                             "the loss function. Note: Setting too low could "
                             "take too long to converge, while setting it too "
                             "high could cause the optimization to diverge.")

    parser.add_argument("--seed", type=int,
                        help="Random seed to for reproducible results.")

    args = parser.parse_args()

    device_name = "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
    device = torch.device(device_name)

    content = utils.load_image(args.content_img, args.max_img_size)
    style = utils.load_image(args.style_img, new_shape=content.shape[2:])

    stylizer = stylize.Stylizer(content, style, args.content_layers, args.style_layers,
                                args.pool_type, device)
    generated = stylizer.stylize(args.init_method, args.content_weight, args.style_weight,
                                 args.content_layer_weights, args.style_layer_weights,
                                 args.max_iter, args.learning_rate)

    utils.save_image(generated, args.output_file)


if __name__ == '__main__':
    main()
