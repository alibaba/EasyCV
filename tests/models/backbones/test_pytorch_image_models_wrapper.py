# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.models.backbones import PytorchImageModelWrapper
from easycv.utils.profiling import benchmark_torch_function


class PytorchImageModelWrapperTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    @torch.no_grad()
    def test_swint_feature(self):
        batch_size = 1
        a = torch.rand(batch_size, 3, 224, 224).to('cuda')

        net = PytorchImageModelWrapper(
            model_name='swin_tiny_patch4_window7_224',
            num_classes=0,
            global_pool='').to('cuda')
        net.eval()

        self.assertTrue(net(a)[-1].size(0) == batch_size)

        self.assertTrue(net(a)[-1].size(1) == 768)

        iter = 100
        t = benchmark_torch_function(iter, net, a)
        print(f'swint extracts feature: {t/batch_size} s/per image')

    @torch.no_grad()
    def test_efficientnet_feature(self):
        batch_size = 1
        a = torch.rand(batch_size, 3, 224, 224).to('cuda')

        net = PytorchImageModelWrapper(
            model_name='efficientnet_b0', num_classes=0,
            global_pool='').to('cuda')
        net.eval()

        self.assertTrue(net(a)[-1].size(0) == batch_size)

        self.assertTrue(net(a)[-1].size(1) == 1280)

        self.assertTrue(net(a)[-1].size(2) == 7)

        self.assertTrue(net(a)[-1].size(3) == 7)

        iter = 100
        t = benchmark_torch_function(iter, net, a)
        print(f'efficient_b0 extracts feature: {t/batch_size} s/per image')

    @torch.no_grad()
    def test_swint_feature_load_modelzoo(self):

        batch_size = 2
        a = torch.rand(batch_size, 3, 224, 224).to('cuda')

        # swin_tiny_patch4_window7_224
        net = PytorchImageModelWrapper(
            model_name='swin_tiny_patch4_window7_224',
            num_classes=0,
            global_pool='').to('cuda')
        net.eval()
        net_feature = net(a)[-1].detach().cpu().numpy()

        net_random_init = PytorchImageModelWrapper(
            model_name='swin_tiny_patch4_window7_224',
            num_classes=0,
            global_pool='').to('cuda')
        net_random_init.eval()
        net_random_init_feature = net_random_init(a)[-1].detach().cpu().numpy()

        self.assertFalse(np.allclose(net_random_init_feature, net_feature))


if __name__ == '__main__':
    unittest.main()
