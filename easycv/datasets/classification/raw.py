# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
from PIL import Image

from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset


@DATASETS.register_module
class ClsDataset(BaseDataset):
    """Dataset for classification

    Args:
        data_source: data source to parse input data
        pipeline: transforms list
    """

    def __init__(self, data_source, pipeline):
        super(ClsDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):

        img, target = self.data_source.get_sample(idx)

        if isinstance(img, list):
            # img is list, means contains multi img
            mlist = []
            for m in img:
                assert isinstance(m, Image.Image), \
                    f'The output from the data source must be an Image, got: {type(m)}. \
                    Please ensure that the list file does not contain labels.'

                m = (self.pipeline(m)).unsqueeze(0)
                mlist.append(m)
            img_cat = torch.cat(mlist, dim=0)
            return dict(img=img_cat, gt_label=torch.tensor(target).long())
        else:
            img = self.pipeline(img)
            return dict(img=img, gt_label=target)

    def evaluate(self, results, evaluators, logger=None, topk=(1, 5)):
        '''evaluate classification task

        Args:
            results: a dict of list of tensor, including prediction and groundtruth
                info, where prediction tensor is NxC，and the same with groundtruth labels.

            evaluators: a list of evaluator

        Return:
            eval_result: a dict of float, different metric values
        '''
        assert len(evaluators) == 1, \
            'classification evaluation only support one evaluator'
        gt_labels = results.pop('gt_labels')
        eval_res = evaluators[0].evaluate(results, gt_labels)

        return eval_res
