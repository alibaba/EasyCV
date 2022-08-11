# Copyright (c) Alibaba, Inc. and its affiliates.
# debug 
import sys
sys.path.append('/root/code/ocr/EasyCV')

from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset

@DATASETS.register_module()
class OCRDetDataset(BaseDataset):
    """Dataset for ocr text detection
    """
    
    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRDetDataset, self).__init__(data_source, pipeline, profiling=profiling)
        
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        data_dict = self.data_source.get_sample(idx)
        data_dict = self.pipeline(data_dict)
        return data_dict
    
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
    cfg = mmcv_config_fromfile('configs/ocr/det_model.py')
    dataset = OCRDetDataset(data_source=cfg.data_source,pipeline=cfg.train_pipeline)
    for i in dataset:
        print(i.keys())
        print(i['threshold_map'].shape)
        print(i['threshold_mask'].shape)
        exit()