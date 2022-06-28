# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
from multiprocessing import cpu_count

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from .base import SegSourceBase


def parse_raw(source_item, classes):
    img_path, seg_path = source_item
    result = {'filename': img_path, 'seg_filename': seg_path}
    return result


@DATASOURCES.register_module
class SegSourceRaw(SegSourceBase):
    """Data source for semantic segmentation.
    data format is as follows:

    ├── data_dir
    │   │   ├── images
    │   │   │   ├── 1.jpg
    │   │   │   ├── 2.jpg
    │   │   │   ├── ...
    │   │   ├── labels
    │   │   │   ├── 1.png
    │   │   │   ├── 2.png
    │   │   │   ├── ...

    Args:
        img_root (str): images dir path
        label_root (str): labels dir path
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_root/label_root will be loaded.
        classes (str | list): classes list or file
        img_suffix (str): image file suffix
        label_suffix (str): label file suffix
        reduce_zero_label (bool): whether to mark label zero as ignored
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            palette of segmentation map, if none, random palette will be generated
        cache_at_init (bool): if set True, will cache in memory in __init__ for faster training
        cache_on_the_fly (bool): if set True, will cache in memroy during training
    """

    def __init__(self,
                 img_root=None,
                 label_root=None,
                 split=None,
                 classes=None,
                 img_suffix='.jpg',
                 label_suffix='.png',
                 reduce_zero_label=False,
                 palette=None,
                 num_processes=int(cpu_count() / 2),
                 cache_at_init=False,
                 cache_on_the_fly=False):

        self.img_root = img_root
        self.label_root = label_root
        self.split = split
        self.classes = classes
        self.PALETTE = palette
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix

        if isinstance(self.img_suffix, str):
            self.img_suffix = [self.img_suffix]
        if isinstance(label_suffix, str):
            self.label_suffix = [self.label_suffix]
        assert isinstance(self.img_suffix, list)
        assert isinstance(self.label_suffix, list)

        super(SegSourceRaw, self).__init__(
            classes=classes,
            reduce_zero_label=reduce_zero_label,
            palette=palette,
            parse_fn=parse_raw,
            num_processes=num_processes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)

    def get_source_iterator(self):
        if self.split is not None:
            with io.open(self.split, 'r') as f:
                lines = f.readlines()

            self.img_files = []
            for line in lines:
                find = False
                for img_suf in self.img_suffix:
                    filename = os.path.join(self.img_root,
                                            line.strip() + img_suf)
                    if os.path.exists(filename):
                        self.img_files.append(filename)
                        find = True
                if not find:
                    logging.warning('Not find file: %s with suffix %s' %
                                    (line, self.img_suffix))
        else:
            self.img_files = [
                os.path.join(self.img_root, i)
                for i in io.listdir(self.img_root, recursive=True)
                if os.path.splitext(i)[-1].lower() in self.img_suffix
            ]

        self.label_files = []
        for img_path in self.img_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            find_label_path = False
            for label_format in self.label_suffix:
                lable_path = os.path.join(self.label_root,
                                          img_name + label_format)
                if io.exists(lable_path):
                    find_label_path = True
                    self.label_files.append(lable_path)
                    break
            if not find_label_path:
                logging.warning(
                    'Not find label file %s for img: %s, skip the sample!' %
                    (lable_path, img_path))
                self.img_files.remove(img_path)

        assert len(self.img_files) == len(self.label_files)
        assert len(
            self.img_files) > 0, 'No samples found in %s' % self.img_root

        return list(zip(self.img_files, self.label_files))
