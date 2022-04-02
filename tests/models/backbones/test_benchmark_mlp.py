# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import BenchMarkMLP


class BenchMarkMLPTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_benchmark_mlp(self):

        model = BenchMarkMLP(feature_num=512)
        model.init_weights()
        model.train()
        feas = torch.randn(2, 512)
        output = model(feas)
        self.assertEqual(output[0].shape, torch.Size([2, 512]))


if __name__ == '__main__':
    unittest.main()
