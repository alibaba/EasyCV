# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np

from easycv.core.evaluation import KeyPointEvaluator


class KeyPointEvaluatorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_keypoint_evaluator_pck(self):
        evaluator = KeyPointEvaluator(pck_thr=0.5, pckh_thr=0.5, auc_nor=30)
        output = np.zeros((5, 3))
        target = np.zeros((5, 3))
        mask = np.zeros((5, 3))
        mask[:, :2] = 1
        # first channel
        output[0] = [10, 0, 0]
        target[0] = [10, 0, 0]
        # second channel
        output[1] = [20, 20, 0]
        target[1] = [10, 10, 0]
        # third channel
        output[2] = [0, 0, 0]
        target[2] = [-1, 0, 0]
        # fourth channel
        output[3] = [30, 30, 0]
        target[3] = [30, 30, 0]
        # fifth channel
        output[4] = [0, 10, 0]
        target[4] = [0, 10, 0]
        preds = {'keypoints': output}
        db = {
            'joints_3d': target,
            'joints_3d_visible': mask,
            'bbox': [10, 10, 10, 10],
            'head_size': 10
        }
        eval_res = evaluator.evaluate([preds, preds], [db, db])
        self.assertAlmostEqual(eval_res['PCK'], 0.8)
        self.assertAlmostEqual(eval_res['PCKh'], 0.8)
        self.assertAlmostEqual(eval_res['EPE'], 3.0284271240234375)
        self.assertAlmostEqual(eval_res['AUC'], 0.86)
        self.assertAlmostEqual(eval_res['NME'], 3.0284271240234375)


if __name__ == '__main__':
    unittest.main()
