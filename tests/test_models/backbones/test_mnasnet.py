# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import random
import unittest

import numpy as np
import torch

from easycv.models import modelzoo
from easycv.models.backbones import MNASNet


class MnasnetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_mnasnet_withfc(self):
        for alpha in [0.5, 0.75, 1.0, 1.3]:
            with torch.no_grad():
                # input data
                batch_size = random.randint(10, 30)
                a = torch.rand(batch_size, 3, 224, 224).to('cuda')

                num_classes = random.randint(10, 1000)
                net = MNASNet(alpha=alpha, num_classes=num_classes).to('cuda')
                net.init_weights()
                net.eval()

                self.assertTrue(len(list(net(a)[-1].shape)) == 2)

                self.assertTrue(net(a)[-1].size(1) == num_classes)

                self.assertTrue(net(a)[-1].size(0) == batch_size)

    def test_mnasnet_withoutfc(self):
        for alpha in [0.5, 0.75, 1.0, 1.3]:
            with torch.no_grad():
                # input data
                batch_size = random.randint(10, 30)
                a = torch.rand(batch_size, 3, 224, 224).to('cuda')

                net = MNASNet(alpha=alpha, num_classes=0).to('cuda')
                net.init_weights()
                net.eval()

                self.assertTrue(net(a)[-1].size(1) == 1280)

                self.assertTrue(net(a)[-1].size(0) == batch_size)

    def test_mnasnet_load_modelzoo(self):
        for alpha in [0.5, 1.0]:
            with torch.no_grad():
                net = MNASNet(alpha=1.0, num_classes=1000).to('cuda')
                original_weight = net.layers[0].weight
                original_weight = copy.deepcopy(
                    original_weight.cpu().data.numpy())

                net.init_weights()
                load_weight = net.layers[0].weight.cpu().data.numpy()

                self.assertFalse(np.allclose(original_weight, load_weight))


if __name__ == '__main__':
    unittest.main()
