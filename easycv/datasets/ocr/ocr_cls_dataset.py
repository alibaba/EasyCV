# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.registry import DATASETS
from .ocr_raw_dataset import OCRRawDataset


@DATASETS.register_module(force=True)
class OCRClsDataset(OCRRawDataset):
    """Dataset for ocr text classification
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRRawDataset, self).__init__(
            data_source, pipeline, profiling=profiling)

    def evaluate(self, results, evaluators, logger=None, **kwargs):
        assert len(evaluators) == 1, \
            'classification evaluation only support one evaluator'
        gt_labels = results.pop('label')
        eval_res = evaluators[0].evaluate(results, gt_labels)

        return eval_res
