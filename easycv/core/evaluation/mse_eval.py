# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


@EVALUATORS.register_module
class MSEEvaluator(Evaluator):
    """ MSEEvaluator evaluator,

  """

    def __init__(self,
                 dataset_name=None,
                 metric_names=['avg_mse'],
                 neck_num=None):
        '''
    '''

        self.metric = 'min'
        self.dataset_name = dataset_name
        self.neck_num = neck_num

        super(MSEEvaluator, self).__init__(dataset_name, metric_names)

    def _evaluate_impl(self, results, gt_label):
        """ Retrival evaluate do the topK retrival, by measuring the distance of every 1 vs other.
        get the topK nearest, and count the match of ID. if Retrival = 1, Miss = 0. Finally average all
        RetrivalRate.
        """
        # first print() is to show shape clearly in multi-process situation. don't comment it
        print()

        if self.neck_num is None:
            try:
                results = results.cuda()
            except:
                results = results['neck'].cuda()
        else:
            results = results['neck_%d_0' % self.neck_num].cuda()

        gt_label = gt_label.cuda()

        # print(results.shape)
        # print(gt_label.shape)

        if results.shape[1] > 1:
            n, c = results.size()
            prob = torch.nn.Softmax(dim=1)(results)

            distribute = torch.arange(0, c).repeat(n, 1).to(results.device)
            results = (distribute * prob).sum(dim=1)

        eval_res = {}
        avg_mse = torch.mean(torch.abs(results - gt_label))
        eval_res['avg_mse'] = avg_mse.item()

        return eval_res


METRICS.register_default_best_metric(MSEEvaluator, 'avg_mse', 'max')
