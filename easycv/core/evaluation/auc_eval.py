# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict

from sklearn.metrics import roc_auc_score

from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


@EVALUATORS.register_module
class AucEvaluator(Evaluator):
    """ AUC evaluator for binary classification only.
  """

    def __init__(self,
                 dataset_name=None,
                 metric_names=['neck_auc'],
                 neck_num=None):
        '''
    Args:
      dataset_name: eval dataset name
      metric_names: eval metrics name
      neck_num: some model contains multi-neck to support multitask, neck_num means use the no.neck_num  neck output of model to eval
    '''
        self.dataset_name = dataset_name
        self.neck_num = neck_num

        super(AucEvaluator, self).__init__(dataset_name, metric_names)

    def _evaluate_impl(self, predictions, gt_labels):
        ''' python evaluation code which will be run after
        all test batched data are predicted

        Args:
            predictions: dict of tensor with shape NxC, from each cls heads
            gt_labels: tensor with shape NxC

        Return:
            a dict,  each key is metric_name, value is metric value
        '''
        eval_res = OrderedDict()
        target = gt_labels.long()

        # if self.neck_num is not None:
        if self.neck_num is None:
            predictions = {'neck': predictions['neck']}
        else:
            predictions = {
                'neck_%d_0' % self.neck_num:
                predictions['neck_%d_0' % self.neck_num]
            }

        for key, scores in predictions.items():
            assert scores.size(0) == target.size(0), \
                'Inconsistent length for results and labels, {} vs {}'.format(
                scores.size(0), target.size(0))
            target = target.cpu().numpy()
            scores = scores.cpu().numpy()[:, 1]
            auc = roc_auc_score(target, scores)
            eval_res['{}_auc'.format(key)] = auc

        return eval_res


METRICS.register_default_best_metric(AucEvaluator, 'neck_auc', 'max')
