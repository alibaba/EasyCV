# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks import TensorboardLoggerHook as _TensorboardLoggerHook


@HOOKS.register_module()
class TensorboardLoggerHookV2(_TensorboardLoggerHook):

    def visualization_log(self, runner):
        visual_results = runner.visualization_buffer.output
        images = visual_results.get('images', [])
        img_metas = visual_results.get('img_metas', [])
        assert len(images) == len(
            img_metas
        ), 'Output `images` and `img_metas` must keep the same length!'

        for i, img in enumerate(images):
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            else:
                assert isinstance(
                    img, torch.Tensor
                ), 'Only support np.ndarray and torch.Tensor type!'

            self.writer.add_image(
                img_metas[i]['ori_filename'],
                img,
                self.get_iter(runner),
                dataformats='HWC')

    @master_only
    def log(self, runner):
        self.visualization_log(runner)
        super(TensorboardLoggerHookV2, self).log(runner)
