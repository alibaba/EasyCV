# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod

import six

from easycv.utils.eval_utils import generate_best_metric_name


class Evaluator(six.with_metaclass(ABCMeta, object)):
    """ Evaluator interface

    """

    def __init__(self, dataset_name=None, metric_names=[]):
        ''' Construct eval ops from tensor

        Args:
          dataset_name (str): dataset name to be evaluated
          metric_names (List[str]): metric names this evaluator will return
        '''
        # define the metric names, should be the same with the keys of dict returned by
        # self.evaluate()
        self._dataset_name = dataset_name
        self._metric_names = metric_names

    def evaluate(self, prediction_dict, groundtruth_dict, **kwargs):
        eval_res = self._evaluate_impl(prediction_dict, groundtruth_dict,
                                       **kwargs)
        if eval_res is None:
            return

        return_res = {}
        best_metric_name = generate_best_metric_name(self.__class__.__name__,
                                                     self._dataset_name,
                                                     self._metric_names)

        # add best metric data to eval result
        if self._metric_names is not None:
            for k in self.metric_names:
                if k in eval_res.keys():
                    for rk in best_metric_name:
                        if k in rk:
                            return_res[rk] = eval_res[k]

        # use dataset_name as metric prefix if assigned
        for k in eval_res.keys():
            if getattr(self, 'dataset_name', None) is not None:
                return_res[self.dataset_name + '_' + k] = eval_res[k]
            else:
                return_res[k] = eval_res[k]

        return return_res

    # want use Evaluator as dummy evaluator
    @abstractmethod
    def _evaluate_impl(self, predictions, labels, **kwargs):
        ''' python evaluation code which will be run after all test batched data are predicted

        Return:
            a dict,  each key is metric_name, value is metric value
        '''
        pass

    @property
    def metric_names(self):
        return self._metric_names
