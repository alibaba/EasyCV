# Copyright (c) Alibaba, Inc. and its affiliates.
from timm.data import Mixup
from timm.data.mixup import mixup_target

from .registry import HOOKS


class BaseCollateHook(object):
    """Collate fn hook when build dataloader.
    Used when you need to process before or after merges a list of samples to form a mini-batch of Tensor(s).
    """

    def __init__(self) -> None:
        pass

    def before_collate(self, batch):
        return batch

    def after_collate(self, batch):
        return batch


@HOOKS.register_module()
class MixupCollateHook(BaseCollateHook):
    """Mixedup data batch, should be used after merges a list of samples to form a mini-batch of Tensor(s).
    """

    def __init__(self, **kwargs):
        self.mixup = Mixup(**kwargs)

    def after_collate(self, results):
        batch_size = results['img'].size()[0]
        assert batch_size % 2 == 0, 'Batch size should be even when using this, but get {}'.format(
            batch_size)
        samples = results['img']
        targets = results['gt_labels']

        if self.mixup.mode == 'elem':
            lam = self.mixup._mix_elem(samples)
        elif self.mixup.mode == 'pair':
            lam = self.mixup._mix_pair(samples)
        else:
            lam = self.mixup._mix_batch(samples)

        device = samples.device
        targets = mixup_target(
            target=targets,
            num_classes=self.mixup.num_classes,
            lam=lam,
            smoothing=self.mixup.label_smoothing,
            device=device)

        results['img'] = samples
        results['gt_labels'] = targets

        return results
