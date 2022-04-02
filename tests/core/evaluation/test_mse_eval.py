# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.core.evaluation import MSEEvaluator


class MSEEvaluatorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_mse_evaluator(self, dataset_name=None):
        evaluator = MSEEvaluator(dataset_name)
        preds = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5], [0.3, 0.4, 0.3],
                          [0.2, 0.1, 0.7]])
        labels = np.array([0, 1, 2, 2])
        preds_dict = {'neck': torch.from_numpy(preds).float()}

        eval_res = evaluator.evaluate(preds_dict, torch.from_numpy(labels))
        self.assertEqual(format(eval_res['avg_mse'], '.4f'), '0.6665')
        self.assertEqual(
            format(eval_res['MSEEvaluator_avg_mse'], '.4f'), '0.6665')

    def test_mse_evaluator_dataset_name(self):
        evaluator = MSEEvaluator('dummy')
        preds = np.array([[0.6, 0.1, 0.3], [0.2, 0.3, 0.5], [0.9, 0.05, 0.05],
                          [0.2, 0.1, 0.7]])
        labels = np.array([0, 0, 1, 2])
        preds_dict = {'neck': torch.from_numpy(preds).float()}

        eval_res = evaluator.evaluate(preds_dict, torch.from_numpy(labels))
        self.assertEqual(format(eval_res['dummy_avg_mse'], '.4f'), '0.7792')
        self.assertEqual(
            format(eval_res['MSEEvaluator_dummy_avg_mse'], '.4f'), '0.7792')


if __name__ == '__main__':
    unittest.main()
