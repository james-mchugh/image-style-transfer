# PyTorch Neural Image Style Transfer
Neural image style transfer is the technique of taking the style from one image
and the content from another and merging them together into a cohesive image.
This is done by using convolutional neural networks to extract representations
of style and content from the respective images. These repesentations are then
used to generate a new image in which the difference between the new image and
these representations is minimized. The code developed here is based on the
work done in:

> L. A. Gatys, A. S. Ecker and M. Bethge, "Image Style Transfer Using
 Convolutional Neural Networks," 2016 IEEE Conference on Computer Vision
> and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 2414-2423.

The code also takes some inspiration from  [Neural Style Tansfer Using
Pytorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) 
by Winston Herring.

## Setup and Run
Currently, the easiest way to get up and running with the image styler is to
utilize the Docker image, but you can also clone the repository if you
desire. A PyPi package will be setup soon!

### Docker
1. Ensure Docker (or NVIDIA-Docker if you want to use a GPU) is installed
2. Run `docker run -it -v $(pwd):/workspace james-mchugh/image-styler` to
get usage information for the container
3. As an example, if you want to use the content from `content.jpg`, the style
from `style.jpg`, and output the image the `output.jpg` run the following
command:
```
docker run -it -v $(pwd):/workspace james-mchugh/image-styler content.jpg
 style.jpg output.jpg
```

### Source
1. Clone the repo
2. In the repository directory, install the Python package using `pip
 install .`
3. Run `python -m image_styler` to get usage information about the package
4. As an example, if you want to use the content from `content.jpg`, the style
from `style.jpg`, and output the image the `output.jpg` run the following
command:
```
python -m image_styler content.jpg style.jpg output.jpg
```

## More examples and details to come!