FROM pytorch/pytorch:latest
ADD . image_styler
RUN pip install ./image_styler && \
    rm -rf image_styler && \
    python -c "import torchvision; torchvision.models.vgg19(True)"

ENTRYPOINT ["python", "-m", "image_styler"]