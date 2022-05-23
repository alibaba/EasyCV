# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

# from easycv.models.utils import accuracy
from easycv.models.utils.accuracy import accuracy


class AccuracyTest(unittest.TestCase):

    def test_2d(self):
        # bacth_size = 5, num_classes = 6
        pred = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6, 0.1],
             [0.3, 0.4, 0.5, 0.6, 0.1, 0.2], [0.4, 0.5, 0.6, 0.1, 0.2, 0.3],
             [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            dtype=torch.float32)
        target1 = torch.tensor([5, 4, 3, 2, 5], dtype=torch.int64)
        target2 = torch.tensor([0, 1, 2, 3, 5], dtype=torch.int64)
        res1 = accuracy(pred, target1, topk=1)
        res2 = accuracy(pred, target2, topk=(1, 5))
        self.assertEqual(len(res1), 1)
        self.assertEqual(int(res1[0].numpy()), 100)
        self.assertEqual(len(res2), 2)
        self.assertEqual(int(res2[0].numpy()), 20)
        self.assertEqual(int(res2[1].numpy()), 60)

    def test_4d(self):
        # bacth_size = 2, image_size = (3, 3), num_classes = 3
        # shape=(bacth_size, num_classes, image_size[0], image_size[1])
        pred = torch.tensor(
            [[[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]],
              [[0.2, 0.2, 0.2], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
              [[0.01, 0.01, 0.01], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]],
             [[[0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]],
              [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
              [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]]]],
            dtype=torch.float32)

        target1 = torch.tensor([[[1, 1, 1], [2, 2, 2], [2, 2, 2]],
                                [[0, 0, 0], [0, 0, 0], [0, 0, 0]]],
                               dtype=torch.int64)
        res1 = accuracy(pred, target1, topk=1)
        self.assertEqual(len(res1), 1)
        self.assertEqual(res1[0], torch.tensor(100.0))

        target2 = torch.tensor([[[1, 1, 1], [255, 255, 255], [2, 2, 2]],
                                [[2, 2, 2], [2, 2, 2], [2, 2, 2]]],
                               dtype=torch.int64)
        res2 = accuracy(pred, target2, topk=(1, 2), ignore_index=255)
        self.assertEqual(len(res2), 2)
        self.assertEqual(res2, [torch.tensor(40.0), torch.tensor(100.0)])

        target3 = torch.tensor([[[255, 255, 255], [2, 2, 2], [0, 0, 0]],
                                [[255, 1, 1], [255, 2, 2], [2, 2, 2]]],
                               dtype=torch.int64)

        res3 = accuracy(pred, target3, topk=(1, 2))
        self.assertEqual(len(res3), 2)
        self.assertAlmostEqual(res3[0].numpy(), 16.6667, delta=0.001)
        self.assertAlmostEqual(res3[1].numpy(), 44.4444, delta=0.001)


if __name__ == '__main__':
    unittest.main()
