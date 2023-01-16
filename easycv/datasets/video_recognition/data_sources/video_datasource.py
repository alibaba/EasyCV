# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os

import torch

from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class VideoDatasource(object):
    """video datasource for video recognition

    This class is used to load video recognition ann_file, return a dict
    containing path of original video or frame dir and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with filepath and label, which are split with a split_char.
    Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3

    Args:
        ann_file (str): Path to the annotation file.
        data_root (str): Data dir for video.
        split (str): Separator used in annotation file.
        multi_class (bool): whether dataset is a multi-class dataset.
        num_classes (int | None): Number of a classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
    """

    def __init__(self,
                 ann_file,
                 data_root=None,
                 split='\t',
                 multi_class=False,
                 num_classes=None,
                 start_index=0,
                 modality='RGB',
                 **kwargs):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.split = split
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.video_infos = self.load_annotations()

    def load_annotations(self):
        """Load annotation file to get video information"""

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split(self.split)
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_root is not None:
                    filename = os.path.join(self.data_root, filename)
                video_infos.append(dict(filename=filename, label=label))
        return video_infos

    def prepare_data(self, idx):

        input_dict = copy.deepcopy(self.video_infos[idx])
        input_dict['start_index'] = self.start_index
        input_dict['modality'] = self.modality
        if self.multi_class and isinstance(input_dict['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[input_dict['label']] = 1.
            input_dict = onehot
        return input_dict

    def __len__(self):
        """Return the length of data infos.
        """
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get item from video infos according to the given indix.
        """
        return self.prepare_data(idx)
