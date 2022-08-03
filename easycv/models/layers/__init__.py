#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import importlib
import inspect
import os

from .base_layer import BaseLayer
from .conv_layer import (ConvLayer, ConvLayer3d, NormActLayer, SeparableConv,
                         TransposeConvLayer)
from .dropout import Dropout, Dropout2d
from .global_pool import GlobalPool
from .identity import Identity
from .linear_attention import LinearSelfAttention
from .linear_layer import GroupLinear, LinearLayer
from .non_linear_layers import get_activation_fn
from .normalization_layers import (AdjustBatchNormMomentum,
                                   get_normalization_layer, norm_layers_tuple)

__all__ = [
    'ConvLayer',
    'ConvLayer3d',
    'SeparableConv',
    'NormActLayer',
    'TransposeConvLayer',
    'LinearLayer',
    'GroupLinear',
    'GlobalPool',
    'Identity',
    'PixelShuffle',
    'UpSample',
    'MaxPool2d',
    'AvgPool2d',
    'Dropout',
    'Dropout2d',
    'SinusoidalPositionalEncoding',
    'LearnablePositionEncoding',
    'AdjustBatchNormMomentum',
    'Flatten',
    'MultiHeadAttention',
    'SingleHeadAttention',
    'Softmax',
    'LinearSelfAttention',
]


# iterate through all classes and fetch layer specific arguments
def layer_specific_args(parser: argparse.ArgumentParser):
    layer_dir = os.path.dirname(__file__)
    parsed_layers = []
    for file in os.listdir(layer_dir):
        path = os.path.join(layer_dir, file)
        if (not file.startswith('_') and not file.startswith('.')
                and (file.endswith('.py') or os.path.isdir(path))):
            layer_name = file[:file.find('.py')] if file.endswith(
                '.py') else file
            module = importlib.import_module('easycv.models.layers.' +
                                             layer_name)
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseLayer) and name not in parsed_layers:
                    parser = cls.add_arguments(parser)
                    parsed_layers.append(name)
    return parser


def arguments_nn_layers(parser: argparse.ArgumentParser):
    # Retrieve layer specific arguments
    parser = layer_specific_args(parser)

    # activation and normalization arguments
    from easycv.models.layers.activation import arguments_activation_fn

    parser = arguments_activation_fn(parser)

    from easycv.models.layers.normalization import arguments_norm_layers

    parser = arguments_norm_layers(parser)

    return parser
