# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import random
import unittest

import numpy as np
import torch

from easycv.models import modelzoo
from easycv.models.backbones.bninception import BNInception


class BNInceptionTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_bninception_withfc(self):
        with torch.no_grad():
            # input data
            batch_size = random.randint(10, 30)
            a = torch.rand(batch_size, 3, 224, 224).to('cuda')

            num_classes = random.randint(10, 1000)
            net = BNInception(num_classes=num_classes).to('cuda')
            net.init_weights()
            net.eval()

            self.assertTrue(len(list(net(a)[-1].shape)) == 2)
            self.assertTrue(net(a)[-1].size(1) == num_classes)
            self.assertTrue(net(a)[-1].size(0) == batch_size)

    def test_bninception_withoutfc(self):
        with torch.no_grad():
            # input data
            batch_size = random.randint(10, 30)
            a = torch.rand(batch_size, 3, 224, 224).to('cuda')

            net = BNInception(num_classes=0).to('cuda')
            net.init_weights()
            net.eval()

            self.assertTrue(net(a)[-1].size(1) == 1024)
            self.assertTrue(net(a)[-1].size(0) == batch_size)

    def test_bninception_load_modelzoo(self):
        with torch.no_grad():
            net = BNInception(num_classes=1000).to('cuda')
            original_weight = net.conv2_3x3.weight
            original_weight = copy.deepcopy(original_weight.cpu().data.numpy())

            net.init_weights()
            load_weight = net.conv2_3x3.weight.cpu().data.numpy()

            self.assertFalse(np.allclose(original_weight, load_weight))


if __name__ == '__main__':
    unittest.main()
