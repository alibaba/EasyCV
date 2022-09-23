# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.registry import DATASETS
from .ocr_raw_dataset import OCRRawDataset


@DATASETS.register_module()
class OCRRecDataset(OCRRawDataset):
    """Dataset for ocr text recognition
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRRecDataset, self).__init__(
            data_source, pipeline, profiling=profiling)

    def evaluate(self, results, evaluators, logger=None, **kwargs):
        assert len(evaluators) == 1, \
            'ocrrec evaluation only support one evaluator'
        preds_text = results.pop('preds_text')
        label_text = results.pop('label_text')
        eval_res = evaluators[0].evaluate(preds_text, label_text)

        return eval_res
