# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np

from easycv.core.bbox import bbox_util


class BboxUtilTest(unittest.TestCase):

    def test_batched_cxcywh2xyxy_with_shape(self):
        # normal
        normalized_cxcywh = np.array([[0.4, 0.6, 0.1, 0.2],
                                      [0.5, 0.4, 0.2, 0.3]])
        h, w = 500, 600
        xyxy = bbox_util.batched_cxcywh2xyxy_with_shape(
            normalized_cxcywh, shape=[h, w])

        target = np.array([[0.35 * w, 0.5 * h, 0.45 * w, 0.7 * h],
                           [0.4 * w, 0.25 * h, 0.6 * w, 0.55 * h]])

        self.assertEqual(xyxy.all(), target.all())

        # out of bounds
        cxcywh_out = np.array([[0.4, 0.6, 0.1, 0.9], [0.8, 0.4, 0.8, 0.3]])
        xyxy_out = bbox_util.batched_cxcywh2xyxy_with_shape(
            cxcywh_out, shape=[h, w])

        target_out = np.array([[0.35 * w, 0.15 * h, 0.45 * w, 1.0 * h],
                               [0.4 * w, 0.25 * h, 1.0 * w, 0.55 * h]])

        self.assertEqual(xyxy_out.all(), target_out.all())


if __name__ == '__main__':
    unittest.main()
