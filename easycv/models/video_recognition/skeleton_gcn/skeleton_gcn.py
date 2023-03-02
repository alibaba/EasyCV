# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) OpenMMLab. All rights reserved.
# Refer to: https://github.com/open-mmlab/mmaction2/blob/master/mmaction/models/skeleton_gcn/skeletongcn.py
from easycv.models.builder import MODELS
from .base import BaseGCN


@MODELS.register_module()
class SkeletonGCN(BaseGCN):
    """Spatial temporal graph convolutional networks."""

    def forward_train(self, skeletons, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        losses = dict()

        x = self.extract_feat(skeletons)
        output = self.cls_head(x)
        gt_labels = labels.squeeze(-1)
        loss = self.cls_head.loss(output, gt_labels)
        losses.update(loss)

        return losses

    def forward_test(self, skeletons, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        result = {'prob': output.cpu()}
        return result
