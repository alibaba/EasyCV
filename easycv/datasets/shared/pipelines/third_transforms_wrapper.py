# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import inspect
from enum import EnumMeta

import torch
from torchvision import transforms as _transforms

from easycv.datasets.registry import PIPELINES


def is_child_of(obj, cls):
    try:
        for i in obj.__bases__:
            if i is cls or isinstance(i, cls):
                return True
        for i in obj.__bases__:
            if is_child_of(i, cls):
                return True
    except AttributeError:
        return is_child_of(obj.__class__, cls)
    return False


def get_args(obj):
    full_args_spec = inspect.getfullargspec(obj)
    args = [] if not full_args_spec.args else full_args_spec.args

    if (args and args[0] in ['self', 'cls']):
        args.pop(0)

    return args


def _reset_forward(obj):
    original_forward = obj.forward

    def _new_forward(self, results):
        img = results['img']
        img = original_forward(self, img)
        results['img'] = img
        return results

    setattr(obj, 'forward', _new_forward)


def _reset_call(obj):
    original_call = obj.__call__

    def _new_call(self, results):
        img = results['img']
        img = original_call(self, img)
        results['img'] = img

        return results

    setattr(obj, '__call__', _new_call)


# TODO: find a more pretty way to wrap third transfomrs or import fixed api to warp
def wrap_torchvision_transforms(transform_obj):
    transform_obj = copy.deepcopy(transform_obj)
    # args_format = ['img', 'pic']
    if is_child_of(transform_obj, torch.nn.Module):
        args = get_args(transform_obj.forward)
        if len(args) == 1:  # and args[0] in args_format:
            _reset_forward(transform_obj)
    elif hasattr(transform_obj, '__call__'):
        args = get_args(transform_obj.__call__)
        if len(args) == 1:  # and args[0] in args_format:
            _reset_call(transform_obj)
    else:
        pass


skip_list = ['Compose', 'RandomApply']
_transforms_names = locals()
# register all existing transforms in torchvision
for member in inspect.getmembers(_transforms, inspect.isclass):
    obj_name, obj = member[0], member[1]
    if obj_name in skip_list:
        continue
    if isinstance(obj, EnumMeta):
        continue
    _transforms_names[obj_name] = type(obj_name, (obj, ), dict())
    wrap_torchvision_transforms(_transforms_names[obj_name])
    PIPELINES.register_module(_transforms_names[obj_name])
