# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import random
import unittest

import numpy as np
import torch

from easycv.models import modelzoo
from easycv.models.backbones import Inception3


class InceptionV3Test(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_inceptionv3_withfc(self):
        with torch.no_grad():
            # input data
            batch_size = random.randint(10, 30)
            a = torch.rand(batch_size, 3, 224, 224).to('cuda')

            num_classes = random.randint(10, 1000)
            net = Inception3(
                aux_logits=True, num_classes=num_classes).to('cuda')
            net.init_weights()
            net.eval()

            self.assertTrue(len(list(net(a)[-1].shape)) == 2)

            self.assertTrue(net(a)[-1].size(1) == num_classes)

            self.assertTrue(net(a)[-1].size(0) == batch_size)

    def test_inceptionv3_withoutfc(self):
        with torch.no_grad():
            # input data
            batch_size = random.randint(10, 30)
            a = torch.rand(batch_size, 3, 224, 224).to('cuda')

            net = Inception3(aux_logits=True, num_classes=0).to('cuda')
            net.init_weights()
            net.eval()

            self.assertTrue(net(a)[-1].size(1) == 2048)

            self.assertTrue(net(a)[-1].size(0) == batch_size)

    def test_inceptionv3_load_modelzoo(self):
        with torch.no_grad():
            net = Inception3(aux_logits=True, num_classes=1000).to('cuda')
            original_weight = net.Conv2d_1a_3x3.conv.weight
            original_weight = copy.deepcopy(original_weight.cpu().data.numpy())

            net.init_weights()
            load_weight = net.Conv2d_1a_3x3.conv.weight.cpu().data.numpy()

            self.assertFalse(np.allclose(original_weight, load_weight))


if __name__ == '__main__':
    unittest.main()
