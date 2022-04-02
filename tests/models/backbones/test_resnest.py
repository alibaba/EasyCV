# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.models.backbones.resnest import ResNeSt
from easycv.utils.profiling import benchmark_torch_function


class ResNeStTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_resnest_withoutfc(self):
        batch_size = 2
        images = torch.rand(batch_size, 3, 224, 224).to('cuda')
        model = ResNeSt(200).to('cuda')
        model.init_weights()
        output = model(images)
        self.assertEqual(output[0].shape, torch.Size([batch_size, 2048, 7, 7]))

    def test_resnest_withfc(self):
        batch_size = 2
        num_classes = 5
        images = torch.rand(batch_size, 3, 224, 224).to('cuda')
        model = ResNeSt(101, num_classes=num_classes).to('cuda')
        model.init_weights()
        output = model(images)
        self.assertEqual(output[0].shape, torch.Size([batch_size,
                                                      num_classes]))

    def test_resnest_jit(self):
        with torch.no_grad():
            # input data
            batch_size = 1
            a = torch.rand(batch_size, 3, 224, 224).to('cuda')

            resnest50 = ResNeSt(50).to('cuda')
            resnest50.init_weights()
            resnest50.eval()

            resnest50_trace = torch.jit.trace(resnest50, a).to('cuda')
            resnest50_trace.eval()

            self.assertTrue(
                np.allclose(
                    resnest50(a)[-1].cpu().data.numpy(),
                    resnest50_trace(a)[-1].cpu().data.numpy(),
                    atol=1e-2))

            resnest50(a)
            iter = 100
            t = benchmark_torch_function(iter, resnest50, a)
            print(f'origin: {t/batch_size} s/per image')

            t = benchmark_torch_function(iter, resnest50_trace, a)
            print(f'trace r50: {t/batch_size} s/per image')


if __name__ == '__main__':
    unittest.main()
