# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.builder import build_model

_model_output_dim = 65536
_base_model_cfg = dict(
    type='DINO',
    pretrained=None,
    train_preprocess=[
        'randomGrayScale', 'gaussianBlur', 'solarize'
    ],  # 2+6 view, has different augment pipeline, dino is complex
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='dynamic_deit_small_p16',
    ),

    # swav need  mulit crop ,doesn't support vit based model
    neck=dict(type='DINONeck', in_dim=384, out_dim=_model_output_dim),
    config=dict(
        # dino head setting
        # momentum_teacher = 0.9995, #0.9995 for batchsize=256
        use_bn_in_head=False,
        norm_last_layer=True,
        drop_path_rate=0.1,
        use_tfrecord_input=False,

        # dino loss settding
        out_dim=_model_output_dim,
        local_crops_number=8,
        warmup_teacher_temp=0.04,  # temperature for sharp softmax
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        epochs=100,
    ))


class DINOTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_dino_train(self):
        model = build_model(_base_model_cfg)
        model.train()
        model.init_before_train()
        batch_size = 3
        imgs = [torch.randn(batch_size, 3, 640, 640)] * 2
        output = model(imgs, mode='train')

        self.assertIn('loss', output)
        self.assertEqual(output['loss'].shape, torch.Size([]))


if __name__ == '__main__':
    unittest.main()
