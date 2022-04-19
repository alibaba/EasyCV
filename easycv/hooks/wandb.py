# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks import WandbLoggerHook as _WandbLoggerHook
from PIL import Image as PILImage


@HOOKS.register_module()
class WandbLoggerHookV2(_WandbLoggerHook):

    def visualization_log(self, runner):
        visual_results = runner.visualization_buffer.output
        images = visual_results.get('images', [])
        img_metas = visual_results.get('img_metas', [])
        assert len(images) == len(
            img_metas
        ), 'Output `images` and `img_metas` must keep the same length!'

        examples = []
        for i, img in enumerate(images):
            assert isinstance(img, np.ndarray)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(img, mode='RGB')
            image = self.wandb.Image(
                pil_image, caption=img_metas[i]['ori_filename'])
            examples.append(image)

        self.wandb.log({'examples': examples})

    @master_only
    def log(self, runner):
        self.visualization_log(runner)
        super(WandbLoggerHookV2, self).log(runner)
