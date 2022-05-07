# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict
import numpy as np
import mmcv
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset
from easycv.core.evaluation.segmentation_eval import eval_metrics, pre_eval_to_metrics
from easycv.utils.logger import print_log
from prettytable import PrettyTable
import warnings


@DATASETS.register_module
class SegDataset(BaseDataset):
    """Dataset for Detection
    """

    def __init__(self, data_source, pipeline, profiling=False):
        """
        Args:
            data_source: Data_source config dict
            pipeline: Pipeline config list
            profiling: If set True, will print pipeline time
            classes: A list of class names, used in evaluation for result and groundtruth visualization
        """

        super(SegDataset, self).__init__(
            data_source, pipeline, profiling=profiling)
        self.num_samples = self.data_source.get_length()
        self.classes = self.data_source.classes

    def __getitem__(self, idx):
        data_dict = self.data_source.get_sample(idx)
        data_dict = self.pipeline(data_dict)
        return data_dict

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.data_source.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            yield results['gt_semantic_seg']

    # def evaluate(self, results, evaluators, logger=None):
    #     pass

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        results = results['seg_pred']
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            num_classes = len(self.classes)

            gt_seg_maps = []
            for i in range(len(self.data_source)):
                sample = self.data_source.get_sample(i)
                gt_seg_maps.append(sample['gt_semantic_seg']) 
            
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.data_source.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.data_source.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.classes is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.classes

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results

