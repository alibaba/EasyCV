# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import copy
from torchvision import transforms as _transforms

from easycv.datasets.registry import PIPELINES


def wrap_torchvision_transforms(transform_obj):
    transform_obj = copy.deepcopy(transform_obj)

    if hasattr(transform_obj, '__call__'):
        original_call = transform_obj.__call__
    else:
        return

    def _new_call(self, results):
        img = results['img']
        img = original_call(self, img)
        results['img'] = img

        return results

    
    setattr(transform_obj, '__call__', _new_call)


skip_list = ['Compose', 'RandomApply']
# register all existing transforms in torchvision
for member in inspect.getmembers(_transforms, inspect.isclass):
    obj_name, obj = member[0], member[1]
    if obj_name in skip_list:
        continue
    wrap_torchvision_transforms(obj)
    PIPELINES.register_module(obj)
