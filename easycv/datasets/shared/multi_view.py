# Copyright (c) Alibaba, Inc. and its affiliates.
import copy

from PIL import Image

from easycv.datasets.builder import build_datasource
from easycv.datasets.registry import DATASETS, PIPELINES
from easycv.datasets.shared.base import BaseDataset
from easycv.datasets.shared.pipelines.transforms import Compose
from easycv.utils.registry import build_from_cfg


@DATASETS.register_module
class MultiViewDataset(BaseDataset):
    """The dataset outputs multiple views of an image.
    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.
    Args:
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines.
    """

    def __init__(self, data_source, num_views, pipelines):
        self.data_source = build_datasource(data_source)

        pipelines_list = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            pipelines_list.append(pipeline)

        self.transforms_list = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            self.transforms_list.extend([pipelines_list[i]] * num_views[i])

    def __getitem__(self, idx):
        results = self.data_source.get_sample(idx)

        img = results['img']
        assert isinstance(img, Image.Image), \
            f'The output from the data source must be an Image, got: {type(img)}. \
            Please ensure that the list file does not contain labels.'

        imgs_list = []
        # only perform transforms to img
        for trans in self.transforms_list:
            tmp_input = {'img': copy.deepcopy(img)}
            tmp_result = trans(tmp_input)
            imgs_list.append(tmp_result['img'])

        results['img'] = imgs_list

        return results

    def evaluate(self, results, evaluators, logger=None):
        raise NotImplementedError
