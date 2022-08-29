# Copyright (c) Alibaba, Inc. and its affiliates.
# debug 
import sys
sys.path.append('/root/code/ocr/EasyCV')

from easycv.datasets.registry import DATASETS
from easycv.datasets.ocr.ocr_det import OCRDetDataset

@DATASETS.register_module(force=True)
class OCRClsDataset(OCRDetDataset):
    """Dataset for ocr text classification
    """
    
    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRDetDataset, self).__init__(data_source, pipeline, profiling=profiling)
    
    def evaluate(self, results, evaluators, logger=None, **kwargs):
        assert len(evaluators) == 1, \
            'classification evaluation only support one evaluator'
        gt_labels = results.pop('label')
        eval_res = evaluators[0].evaluate(results, gt_labels)

        return eval_res
    
if __name__ == "__main__":
    from easycv.utils.config_tools import mmcv_config_fromfile
    cfg = mmcv_config_fromfile('configs/ocr/direction_model.py')
    dataset = OCRClsDataset(data_source=cfg.val_dataset.data_source, pipeline=cfg.val_pipeline)
    for member in dataset:
        print(member.keys())