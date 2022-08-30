# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.ocr.ocr_det_dataset import OCRDetDataset
from easycv.datasets.registry import DATASETS


@DATASETS.register_module(force=True)
class OCRClsDataset(OCRDetDataset):
    """Dataset for ocr text classification
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRDetDataset, self).__init__(
            data_source, pipeline, profiling=profiling)

    def evaluate(self, results, evaluators, logger=None, **kwargs):
        assert len(evaluators) == 1, \
            'classification evaluation only support one evaluator'
        gt_labels = results.pop('label')
        eval_res = evaluators[0].evaluate(results, gt_labels)

        return eval_res
