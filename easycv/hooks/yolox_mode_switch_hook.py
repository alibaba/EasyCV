# Copyright (c) Alibaba, Inc. and its affiliates.
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import Hook

from .registry import HOOKS


@HOOKS.register_module()
class YOLOXModeSwitchHook(Hook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    Args:
        no_aug_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Default: 15.
       skip_type_keys (list[str], optional): Sequence of type string to be
            skip pipeline. Default: ('Mosaic', 'RandomAffine', 'MixUp')
    """

    def __init__(self,
                 no_aug_epochs=15,
                 skip_type_keys=('MMMosaic', 'MMRandomAffine', 'MMMixUp'),
                 **kwargs):
        super(YOLOXModeSwitchHook, self).__init__()
        self.no_aug_epochs = no_aug_epochs
        self.skip_type_keys = skip_type_keys

    def before_train_epoch(self, runner):
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        if (epoch + 1) == runner.max_epochs - self.no_aug_epochs:
            runner.logger.info('No mosaic and mixup aug now!')
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            runner.logger.info('Add additional L1 loss now!')
            model.head.use_l1 = True
