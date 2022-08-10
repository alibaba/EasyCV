# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from collections.abc import Sequence

from easycv.datasets.registry import PIPELINES
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
