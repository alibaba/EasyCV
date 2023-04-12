# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.core.evaluation import ClsEvaluator


class ClassificationEvaluatorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_classification_evaluator(self):
        evaluator = ClsEvaluator()

        preds = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.2, 0.5, 0.1, 0.1, 0.1],
            [0.8, 0.0, 0.0, 0.1, 0.1],
            [0.7, 0.2, 0.1, 0.0, 0.0],
        ])
        labels = np.array([1, 1, 0, 2])
        preds_dict = {'neck': torch.from_numpy(preds)}
        eval_res = evaluator.evaluate(preds_dict, torch.from_numpy(labels))
        self.assertAlmostEqual(eval_res['neck_top1'], 75)
        self.assertAlmostEqual(eval_res['neck_top5'], 100)


if __name__ == '__main__':
    unittest.main()
