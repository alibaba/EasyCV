# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks import TensorboardLoggerHook as _TensorboardLoggerHook


@HOOKS.register_module()
class TensorboardLoggerHookV2(_TensorboardLoggerHook):

    def visualization_log(self, runner):
        """Images Visulization.
        `visualization_buffer` is a dictionary containing:
            images (list): list of visulaized images.
            img_metas (list of dict, optional): dict containing ori_filename and so on.
                ori_filename will be displayed as the tag of the image by default.
        """
        visual_results = runner.visualization_buffer.output
        for vis_key, vis_result in visual_results.items():
            images = vis_result.get('images', [])
            img_metas = vis_result.get('img_metas', None)
            if img_metas is not None:
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

                default_name = 'image_%i' % i
                filename = img_metas[i].get(
                    'ori_filename',
                    default_name) if img_metas is not None else default_name
                self.writer.add_image(
                    f'{vis_key}/{filename}',
                    img,
                    self.get_iter(runner),
                    dataformats='HWC')

    @master_only
    def log(self, runner):
        self.visualization_log(runner)
        super(TensorboardLoggerHookV2, self).log(runner)

    def after_train_iter(self, runner):
        super(TensorboardLoggerHookV2, self).after_train_iter(runner)
        # clear visualization_buffer after each iter to ensure that it is only written once,
        # avoiding repeated writing of the same image buffer every self.interval
        runner.visualization_buffer.clear_output()
