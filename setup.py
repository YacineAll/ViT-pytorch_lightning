#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.0.0',
    description='ViT transformer with pytorch lightning',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/YacineAll/ViT-pytorch_lightning',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

