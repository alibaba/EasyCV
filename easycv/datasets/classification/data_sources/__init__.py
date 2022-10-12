# Copyright (c) Alibaba, Inc. and its affiliates.
from .cifar import ClsSourceCifar10, ClsSourceCifar100
from .class_list import ClsSourceImageListByClass
from .cub import ClsSourceCUB
from .image_list import ClsSourceImageList
from .imagenet_tfrecord import ClsSourceImageNetTFRecord
from .imagenet import ClsSourceImageNet1k

__all__ = [
    'ClsSourceCifar10', 'ClsSourceCifar100', 'ClsSourceImageListByClass',
    'ClsSourceImageList', 'ClsSourceImageNetTFRecord', 'ClsSourceCUB', 'ClsSourceImageNet1k'
]
