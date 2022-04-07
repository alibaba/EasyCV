# Copyright (c) Alibaba, Inc. and its affiliates.
from PIL import Image

from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset


@DATASETS.register_module
class ClsOdpsDataset(BaseDataset):
    """Dataset for rotation prediction
    """

    def __init__(self,
                 data_source,
                 pipeline,
                 image_key='url_image',
                 label_key='label',
                 **kwargs):
        super(ClsOdpsDataset, self).__init__(data_source, pipeline)
        self.image_key = image_key
        self.label_key = label_key

    def __getitem__(self, idx):
        data_dict = self.data_source.get_sample(idx)
        assert (self.image_key in data_dict.keys())
        assert (self.label_key in data_dict.keys())

        img = data_dict[self.image_key]
        label = int(data_dict[self.label_key])
        assert isinstance(img, Image.Image), \
            f'The output from the data source must be an Image, got: {type(img)}. \
            Please ensure that the list file does not contain labels.'

        img = self.pipeline(img)
        return dict(img=img, gt_labels=label)

    def evaluate(self, results, evaluators, logger=None):
        raise NotImplementedError
