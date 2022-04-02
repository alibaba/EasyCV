# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.core.evaluation import RetrivalTopKEvaluator


class RetrivalTopKEvaluatorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_retrival_topk_evaluator_top_1_2(self):
        evaluator = RetrivalTopKEvaluator(topk=(1, 2))
        preds = np.array([[0.6, 0.1, 0.3], [0.2, 0.3, 0.5], [0.9, 0.05, 0.05],
                          [0.2, 0.1, 0.7]])
        labels = np.array([0, 0, 1, 2])

        preds_dict = {'neck': torch.from_numpy(preds).float()}
        eval_res = evaluator.evaluate(preds_dict, torch.from_numpy(labels))

        self.assertEqual(eval_res['R@K=1'], 0.0)
        self.assertEqual(eval_res['R@K=2'], 50.0)
        self.assertEqual(eval_res['RetrivalTopKEvaluator_R@K=1'], 0.0)

    def test_retrival_topk_evaluator(self):
        evaluator = RetrivalTopKEvaluator(topk=(1, 2))
        cls_num = 10
        batch_size = 32
        np.random.seed(2022)
        preds = np.random.rand(batch_size, cls_num)
        labels = np.random.randint(cls_num, size=(batch_size))
        preds_dict = {'neck': torch.from_numpy(preds).float()}
        eval_res = evaluator.evaluate(preds_dict, torch.from_numpy(labels))

        self.assertEqual(eval_res['R@K=1'], 15.625)
        self.assertEqual(eval_res['R@K=2'], 21.875)
        self.assertEqual(eval_res['RetrivalTopKEvaluator_R@K=1'], 15.625)


if __name__ == '__main__':
    unittest.main()
