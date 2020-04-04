"""Generate a stylized image based on the provided images.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import argparse

# Third party imports
import torch
import numpy as np

# Local application imports
from . import utils, models, loss

# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------

DEFAULT_CONTENT_LAYERS = ["conv4_2"]
DEFAULT_CONTENT_LAYER_WEIGHTS = [1]

DEFAULT_STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
DEFAULT_STYLE_LAYER_WEIGHTS = [1/len(DEFAULT_STYLE_LAYERS)] * \
                              len(DEFAULT_STYLE_LAYERS)

DEFAULT_MAX_IMAGE_SIZE = 224
DEFAULT_NUM_ITER = 1000

DEFAULT_CONTENT_STYLE_WEIGHT_RATIO = 1/10**-3


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("content_img",
                        help="path to the content image to stylize")
    parser.add_argument("style_img",
                        help="path to the style image to transfer style from")
    parser.add_argument("output_file",
                        help="path to output file")
    parser.add_argument("--gpu",
                        help="use the GPU (if available)",
                        action="store_true")

    args = parser.parse_args()

    device_name = "cuda:0" if torch.cuda.is_available() and args.gpu else "cpu"
    device = torch.device(device_name)

    content_array = utils.load_image(args.content_img, DEFAULT_MAX_IMAGE_SIZE)
    style_array = utils.load_image(args.style_img, DEFAULT_MAX_IMAGE_SIZE)

    content = torch.as_tensor(content_array, dtype=torch.float32,
                              device=device)
    style = torch.as_tensor(style_array, dtype=torch.float32, device=device)
    generated = torch.as_tensor(content_array, dtype=torch.float32,
                                device=device)
    generated.requires_grad = True

    model: models.VGG = models.vgg(19)
    model.to(device)

    content_weight = 1
    style_weight = content_weight / DEFAULT_CONTENT_STYLE_WEIGHT_RATIO

    content_loss_func = loss.ContentLoss(content_weight)
    style_loss_func = loss.StyleLoss(style_weight)

    optimizer = torch.optim.Adam((generated,), lr=100.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     (100, 200, 300, 500),
                                                     gamma=0.5)
    for i in range(DEFAULT_NUM_ITER):
        optimizer.zero_grad()

        _, gen_reprs = model(generated, watch_layers=True)
        _, style_reprs = model(style, watch_layers=True)
        _, content_reprs = model(content, watch_layers=True)
        content_loss = torch.tensor(0.0, device=device)

        for layer, weight in zip(DEFAULT_CONTENT_LAYERS,
                                 DEFAULT_CONTENT_LAYER_WEIGHTS):

            content_loss += weight * content_loss_func(
                content_reprs[layer], gen_reprs[layer], weight)

        for layer, weight in zip(DEFAULT_STYLE_LAYERS,
                                 DEFAULT_STYLE_LAYER_WEIGHTS):
            content_loss += weight * style_loss_func(style_reprs[layer],
                                                     gen_reprs[layer],
                                                     weight)

        if i and i % 100 == 0:
            print(f"Iter: {i} - Loss: {str(content_loss)}")

        content_loss.backward()
        optimizer.step()
        scheduler.step(i)

    utils.save_image(generated.cpu().detach().numpy(), args.output_file)


if __name__ == '__main__':
    main()
