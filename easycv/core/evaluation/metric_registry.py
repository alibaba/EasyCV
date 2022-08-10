# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect


class MetricRegistry(object):

    def __init__(self):
        self.best_metrics = {}

    def get(self, evaluator_type):
        return self.best_metrics[evaluator_type]

    def register_default_best_metric(self,
                                     cls,
                                     metric_name,
                                     metric_cmp_op='max'):
        """ Register default best metric for each evaluator

        Args:
            cls (object):  class object
            metric_name (str or List[str]): default best metric name
            metric_cmp_op (str or List[str]):  metric compare operation, should be one of ["max", "min"]
        """
        if not inspect.isclass(cls):
            raise TypeError('module must be a class, but got {}'.format(
                type(cls)))
        module_name = cls.__name__
        if module_name in self.best_metrics:
            raise KeyError(
                'Default best metrics for {} is already registered'.format(
                    module_name))

        if isinstance(metric_name, str):
            metric_name = [metric_name]

        if isinstance(metric_cmp_op, str):
            if len(metric_name) > 1:
                metric_cmp_op = [
                    metric_cmp_op for i in range(len(metric_name))
                ]
            else:
                metric_cmp_op = [metric_cmp_op]

        assert len(metric_name) == len(
            metric_cmp_op
        ), 'metric_name should be the same length of metric_cmp_op'

        self.best_metrics[module_name] = {
            'metric_name': metric_name,
            'metric_cmp_op': metric_cmp_op
        }
        return cls


METRICS = MetricRegistry()
