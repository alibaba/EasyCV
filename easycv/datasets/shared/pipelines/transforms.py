# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from collections.abc import Sequence

import numpy as np

from easycv.datasets.registry import PIPELINES
from easycv.file.image import load_image
from easycv.framework.errors import TypeError
from easycv.utils.registry import build_from_cfg


@PIPELINES.register_module()
class Compose(object):
    """Compose a data pipeline with a sequence of transforms.
    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms, profiling=False):
        assert isinstance(transforms, Sequence)
        self.profiling = profiling
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            if self.profiling:
                start = time.time()

            data = t(data)

            if self.profiling:
                print(f'{t} time {time.time()-start}')

            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class LoadImage:
    """Load an image from file or numpy or PIL object.
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, to_float32=False, mode='bgr'):
        self.to_float32 = to_float32
        self.mode = mode

    def __call__(self, results):
        """Call functions to load image and get image meta information.
        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results.get('filename', None)
        img = results.get('img', None)

        if img is not None:
            if not isinstance(img, np.ndarray):
                img = np.asarray(img, dtype=np.uint8)
        else:
            assert filename is not None, 'Please provide "filename" or "img"!'
            img = load_image(filename, mode=self.mode)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['ori_img_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"mode='{self.mode}'")

        return repr_str
