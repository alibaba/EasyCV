# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import random
import unittest

import numpy as np
import torch

from easycv.models import modelzoo
from easycv.models.backbones import PlainNet


class PlainNetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_genet_withfc(self):
        for struct in ['normal', 'large']:
            with torch.no_grad():
                # input data
                batch_size = random.randint(10, 30)
                a = torch.rand(batch_size, 3, 224, 224).to('cuda')

                num_classes = random.randint(10, 1000)
                net = PlainNet(struct, num_classes=num_classes).to('cuda')
                net.init_weights()
                net.eval()

                self.assertTrue(len(list(net(a)[-1].shape)) == 2)

                self.assertTrue(net(a)[-1].size(1) == num_classes)

                self.assertTrue(net(a)[-1].size(0) == batch_size)

    def test_genet_withoutfc(self):
        for struct in ['normal', 'large']:
            with torch.no_grad():
                # input data
                batch_size = random.randint(10, 30)
                a = torch.rand(batch_size, 3, 256, 256).to('cuda')

                net = PlainNet(struct, num_classes=0).to('cuda')
                net.init_weights()
                net.eval()

                self.assertTrue(net(a)[-1].size(1) == 2560)

                self.assertTrue(net(a)[-1].size(0) == batch_size)

    def test_genet_load_modelzoo(self):
        for struct in ['normal', 'large']:
            with torch.no_grad():
                net = PlainNet(struct, num_classes=1000).to('cuda')
                original_weight = net.module_list[0].netblock.weight
                original_weight = copy.deepcopy(
                    original_weight.cpu().data.numpy())

                net.init_weights(net.pretrained)
                load_weight = net.module_list[0].netblock.weight.cpu(
                ).data.numpy()

                self.assertFalse(np.allclose(original_weight, load_weight))

                self.assertTrue(net.pretrained == modelzoo.genet['PlainNet' +
                                                                 struct])


if __name__ == '__main__':
    unittest.main()
