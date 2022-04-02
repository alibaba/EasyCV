# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import time

import numpy as np
# not useful in this CR, future support albumentations
# import mkl
# mkl.get_max_threads()
from albumentations import (CLAHE, Blur, Flip, GaussNoise, GridDistortion,
                            HorizontalFlip, HueSaturationValue, IAAEmboss,
                            IAAPerspective, IAAPiecewiseAffine, IAASharpen,
                            MedianBlur, MotionBlur, OneOf, OpticalDistortion,
                            RandomBrightness, RandomContrast, RandomRotate90,
                            ShiftScaleRotate, Transpose)
from PIL import Image
from torchvision import transforms as _transforms

from easycv.datasets.registry import PIPELINES

albumentation_list = [
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion,
    HueSaturationValue, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf
]

PIPELINES.register_module(ShiftScaleRotate)
PIPELINES.register_module(GaussNoise)
PIPELINES.register_module(MotionBlur)

# register all existing transforms in torchvision
for m in inspect.getmembers(_transforms, inspect.isclass):
    # use self-implement Compose
    if m[0] == 'Compose':
        continue
    PIPELINES.register_module(m[1])


@PIPELINES.register_module
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, profiling=False):
        self.transforms = transforms
        self.profiling = profiling

    def __call__(self, img):
        for t in self.transforms:
            if self.profiling:
                start = time.time()

            if isinstance(t, tuple(albumentation_list)):
                img_np = np.array(img)
                augmented = t(image=img_np)
                img = Image.fromarray(augmented['image'])
            else:
                img = t(img)

            if self.profiling:
                print(f'{t} time {time.time()-start}')
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
