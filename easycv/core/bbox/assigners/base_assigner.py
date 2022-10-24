# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod


class BaseBBoxAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""
