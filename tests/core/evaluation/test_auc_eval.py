import unittest

import numpy as np
import torch

from easycv.core.evaluation import AucEvaluator


class AucEvaluatorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_auc_evaluator(self):
        evaluator = AucEvaluator()
        preds = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])
        labels = np.array([0, 1, 1, 0])
        preds_dict = {'neck': torch.from_numpy(preds)}
        eval_res = evaluator.evaluate(preds_dict, torch.from_numpy(labels))
        self.assertEqual(eval_res['neck_auc'], 0.75)
        self.assertEqual(eval_res['AucEvaluator_neck_auc'], 0.75)


if __name__ == '__main__':
    unittest.main()
