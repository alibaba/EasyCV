# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math

import torch
from mmcv.runner import Hook

from easycv.utils import dist_utils, py_util
from .registry import HOOKS


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.

    In Yolo5s, ema help increase mAP from 0.27 to 0.353
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.model = copy.deepcopy(
            model.module if dist_utils.is_parallel(model) else model).eval(
            )  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.model.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(
            -x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if dist_utils.is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self,
                    model,
                    include=(),
                    exclude=('process_group', 'reducer')):
        # Update EMA attributes
        py_util.copy_attr(self.model, model, include, exclude)


@HOOKS.register_module
class EMAHook(Hook):
    """ Hook to carry out Exponential Moving Average
    """

    def __init__(self, decay=0.9999, copy_model_attr=()):
        """
        Args:
            decay: decay rate for exponetial moving average
            copy_model_attr:  attribute to copy from origin model to ema model
        """
        self.decay = decay
        self._copy_model_attr = copy_model_attr
        self._init_updates = False

    def before_run(self, runner):
        runner.ema = ModelEMA(runner.model, decay=self.decay)

    def before_train_epoch(self, runner):
        if not self._init_updates:
            runner.ema.updates = runner.iter
            self._init_updates = True

    def after_train_iter(self, runner):
        runner.ema.update(runner.model)
