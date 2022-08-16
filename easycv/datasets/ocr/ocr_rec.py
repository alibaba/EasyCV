# Copyright (c) Alibaba, Inc. and its affiliates.
# debug 
import sys
sys.path.append('/root/code/ocr/EasyCV')

from easycv.datasets.registry import DATASETS
from easycv.datasets.ocr.ocr_det import OCRDetDataset


@DATASETS.register_module()
class OCRRecDataset(OCRDetDataset):
    """Dataset for ocr text recognition
    """
    
    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRRecDataset, self).__init__(data_source, pipeline, profiling=profiling)
    
    def evaluate(self, results, evaluators, logger=None, **kwargs):
        assert len(evaluators) == 1, \
            'ocrdet evaluation only support one evaluator'
        points = results.pop('points')
        ignore_tags = results.pop('ignore_tags')
        polys = results.pop('polys')
        eval_res = evaluators[0].evaluate(points, polys, ignore_tags)

        return eval_res
    

if __name__ == "__main__":
    from easycv.utils.config_tools import mmcv_config_fromfile
    cfg = mmcv_config_fromfile('configs/ocr/rec_model.py')
    print(cfg)
    dataset = OCRRecDataset(data_source=cfg.data_source, pipeline=cfg.train_pipeline)
    for i in dataset:
        print(i.keys())