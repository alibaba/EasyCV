# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import numpy as np
import torch

from easycv.models import Classification
from easycv.utils.test_util import get_tmp_dir


class ClassificationTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    # def test_jit_trace(self):
    #   backbone = dict(type='ResNet', depth=50, out_indices=[4], norm_cfg=dict(type='SyncBN'))
    #   head = dict(type='ClsHead', with_avg_pool=True, in_channels=2048, num_classes=1000)
    #   with torch.no_grad():
    #     batch_size = 1
    #     a = torch.rand(batch_size, 3, 224, 224).to('cuda')

    #     model = Classification(backbone=backbone, head=head)
    #     trace_model = torch.jit.trace(model.forward_test, a)

    def test_jit_script(self):
        # backbone = dict(type='ResNetJIT', depth=50, out_indices=[4], norm_cfg=dict(type='SyncBN'))
        # SyncBN can not support jit
        # error: Could not cast value of type None to bool
        backbone = dict(
            type='ResNetJIT',
            depth=50,
            out_indices=[4],
            norm_cfg=dict(type='BN'))
        head = dict(
            type='ClsHead',
            with_avg_pool=True,
            in_channels=2048,
            num_classes=1000)
        with torch.no_grad():
            batch_size = 1
            a = torch.rand(batch_size, 3, 224, 224).to('cuda')

            model = Classification(backbone=backbone, head=head).to('cuda')
            model.eval()
            model_jit = torch.jit.script(model)

            out_a = model(a, mode='test')
            out_b = model_jit(a, mode='test')
            self.assertTrue(
                np.allclose(
                    out_a['prob'].numpy(), out_b['prob'].numpy(), atol=1e-3))
            self.assertTrue(
                np.allclose(out_a['class'].numpy(), out_b['class'].numpy()))

            out_a = model(a, mode='extract')
            out_b = model_jit(a, mode='extract')
            self.assertTrue(
                np.allclose(
                    out_a['neck'].numpy(), out_b['neck'].numpy(), atol=1e-3))

            tmp_dir = get_tmp_dir()
            result_f = os.path.join(tmp_dir, 'model.pt.jit')
            print(f'save jit model to {result_f}')
            torch.jit.save(model_jit, result_f)

            model_load = torch.jit.load(result_f)
            out_c = model_jit(a, mode='extract')
            self.assertTrue(
                np.allclose(
                    out_a['neck'].numpy(), out_c['neck'].numpy(), atol=1e-3))


if __name__ == '__main__':
    unittest.main()
