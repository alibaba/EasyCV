# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.models.heads import ClsHead


class ClsHeadTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_jit_script(self):
        head = ClsHead(
            with_avg_pool=True, in_channels=2048, num_classes=1000).to('cuda')
        head_jit = torch.jit.script(head)

        batch_size = 1
        a = [torch.rand(batch_size, 2048).to('cuda')]
        self.assertTrue(
            np.allclose(
                head(a)[0].cpu().data.numpy(),
                head_jit(a)[0].cpu().data.numpy()))


if __name__ == '__main__':
    unittest.main()
