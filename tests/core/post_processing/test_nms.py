# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np

from easycv.models.pose.utils import oks_iou, oks_nms, soft_oks_nms


class NMSTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_soft_oks_nms(self):
        oks_thr = 0.9
        kpts = []
        kpts.append({
            'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
            'area': 100,
            'score': 0.9
        })
        kpts.append({
            'keypoints': np.tile(np.array([10, 10, 0.9]), [17, 1]),
            'area': 100,
            'score': 0.4
        })
        kpts.append({
            'keypoints': np.tile(np.array([100, 100, 0.9]), [17, 1]),
            'area': 100,
            'score': 0.7
        })

        keep = soft_oks_nms([kpts[i] for i in range(len(kpts))], oks_thr)
        self.assertEqual(keep.all(), np.array([0, 2, 1]).all())

        keep = oks_nms([kpts[i] for i in range(len(kpts))], oks_thr)
        self.assertEqual(keep.all(), np.array([0, 2]).all())

    def test_oks_iou(self):
        result = oks_iou(np.ones([17 * 3]), np.ones([1, 17 * 3]), 1, [1])
        self.assertEqual(result[0], 1.)
        result = oks_iou(np.zeros([17 * 3]), np.ones([1, 17 * 3]), 1, [1])
        self.assertTrue(result[0] < 0.01)


if __name__ == '__main__':
    unittest.main()
