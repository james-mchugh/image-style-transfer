from setuptools import setup, find_packages


def get_requirements():
    with open("requirements.txt") as f:
        requirements = [line.strip() for line in f]

    return requirements


setup(
    name='image-style-transfer',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/james-mchugh/image-style-transfer',
    license='',
    author='jmchugh',
    author_email='',
    description='Experimenting with VGG19 and style transfers in PyTorch.',
    install_requires=get_requirements()
)
