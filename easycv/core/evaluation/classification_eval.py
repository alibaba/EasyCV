# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict

from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


@EVALUATORS.register_module
class ClsEvaluator(Evaluator):
    """ Classification evaluator.
  """

    def __init__(self,
                 topk=(1, 5),
                 dataset_name=None,
                 metric_names=['neck_top1'],
                 neck_num=None):
        '''

        Args:
            top_k: tuple of int, evaluate top_k acc
            dataset_name: eval dataset name
            metric_names: eval metrics name
            neck_num: some model contains multi-neck to support multitask, neck_num means use the no.neck_num  neck output of model to eval
        '''
        self._topk = topk
        self.dataset_name = dataset_name
        self.neck_num = neck_num

        super(ClsEvaluator, self).__init__(dataset_name, metric_names)

    def _evaluate_impl(self, predictions, gt_labels):
        ''' python evaluation code which will be run after all test batched data are predicted

        Args:
            predictions: dict of tensor with shape NxC, from each cls heads
            gt_labels: int32 tensor with shape N

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
            num = scores.size(0)
            _, pred = scores.topk(
                max(self._topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
            for k in self._topk:
                # use contiguous() to avoid eval view failed
                correct_k = correct[:k].contiguous().view(-1).float().sum(
                    0).item()
                acc = correct_k * 100.0 / num
                eval_res['{}_top{}'.format(key, k)] = acc

        return eval_res


METRICS.register_default_best_metric(ClsEvaluator, 'neck_top1', 'max')
