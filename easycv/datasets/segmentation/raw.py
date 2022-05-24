# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset
from easycv.datasets.shared.data_sources.concat import SourceConcat


@DATASETS.register_module
class SegDataset(BaseDataset):
    """Dataset for segmentation
    """

    def __init__(self,
                 data_source,
                 pipeline,
                 ignore_index=255,
                 profiling=False):
        """
        Args:
            data_source: Data_source config dict
            pipeline: Pipeline config list
            ignore_index (int): label index to be ignored
            profiling: If set True, will print pipeline time
        """

        super(SegDataset, self).__init__(
            data_source, pipeline, profiling=profiling)
        self.num_samples = self.data_source.get_length()

        if isinstance(self.data_source, SourceConcat):
            self.classes = self.data_source.data_sources[0].classes
            assert self.data_source.data_sources[
                0].classes == self.data_source.data_sources[1].classes

        self.ignore_index = ignore_index

    def __getitem__(self, idx):
        data_dict = self.data_source.get_sample(idx)
        data_dict = self.pipeline(data_dict)
        return data_dict

    def evaluate(self, results, evaluators=[], **kwargs):
        """Evaluate the dataset.

        Args:
            results: A dict of k-v pair, each v is a list of
                tensor or numpy array for segmentation result. A dictionary containing
                seg_pred: List of length number of test images, integer numpy array of shape
                    [width * height].
            evaluators: evaluators to calculate metric with results and groundtruth_dict
        Returns:
            dict[str, float]: Default metrics.
        """
        predict_segs = results['seg_pred']
        gt_seg_maps = []
        for i in range(len(self.data_source)):
            sample = self.data_source.get_sample(i)

            gt_seg = sample['gt_semantic_seg']
            mask = (gt_seg != self.ignore_index)
            predict_segs[i] = predict_segs[i][mask]
            gt_seg = gt_seg[mask]

            gt_seg_maps.append(gt_seg)

        groundtruth_dict = {'gt_seg_maps': gt_seg_maps}
        results['seg_pred'] = predict_segs

        eval_result = {}
        for evaluator in evaluators:
            eval_result.update(evaluator.evaluate(results, groundtruth_dict))

        return eval_result
