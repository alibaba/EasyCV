# Copyright (c) Alibaba, Inc. and its affiliates.
from .caltech import ClsSourceCaltech101, ClsSourceCaltech256
from .cifar import ClsSourceCifar10, ClsSourceCifar100
from .class_list import ClsSourceImageListByClass
from .cub import ClsSourceCUB
from .flower import ClsSourceFlowers102
from .image_list import ClsSourceImageList, ClsSourceItag
from .imagenet import ClsSourceImageNet1k
from .imagenet_tfrecord import ClsSourceImageNetTFRecord
from .mnist import ClsSourceFashionMnist, ClsSourceMnist

__all__ = [
    'ClsSourceCifar10', 'ClsSourceCifar100', 'ClsSourceImageListByClass',
    'ClsSourceImageList', 'ClsSourceItag', 'ClsSourceImageNetTFRecord',
    'ClsSourceCUB', 'ClsSourceImageNet1k', 'ClsSourceCaltech101',
    'ClsSourceCaltech256', 'ClsSourceFlowers102', 'ClsSourceMnist',
    'ClsSourceFashionMnist'
]
