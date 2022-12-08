# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import logging
import os
from multiprocessing import Pool, cpu_count

import cv2
import mmcv
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from easycv.file.image import load_image as _load_img
from .base import SegSourceBase
from .raw import parse_raw


@DATASOURCES.register_module
class SegSourceCocoStuff10k(SegSourceBase):

    CLASSES = [
        'unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
        'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
        'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
        'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
        'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
        'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood'
    ]
    """
    ```
    data format is as follows:

    ├── data
    │   ├── images
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   ├── ...
    │   ├── annotations
    │   │   ├── 1.mat
    │   │   ├── 2.mat
    │   │   ├── ...
    |   |—— imageLists
    |   |—— |—— train.txt
    │   │   ├── ...
    ```
    Example1:
        data_source = SegSourceCocoStuff10k(
            path='/your/data/imageLists/train.txt',
            label_root='/your/data/annotation',
            img_root='/your/data/images',
            classes=${CLASSES}
        )
    Args:
        path: annotation file
        img_root (str): images dir path
        label_root (str): labels dir path
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
                 path,
                 img_root=None,
                 label_root=None,
                 classes=CLASSES,
                 img_suffix='.jpg',
                 label_suffix='.mat',
                 reduce_zero_label=False,
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 palette=None,
                 num_processes=int(cpu_count() / 2)):

        if classes is not None:
            self.CLASSES = classes
        if palette is not None:
            self.PALETTE = palette

        self.path = path
        self.img_root = img_root
        self.label_root = label_root

        self.img_suffix = img_suffix
        self.label_suffix = label_suffix

        self.reduce_zero_label = reduce_zero_label
        self.cache_at_init = cache_at_init
        self.cache_on_the_fly = cache_on_the_fly
        self.num_processes = num_processes

        if self.cache_at_init and self.cache_on_the_fly:
            raise ValueError(
                'Only one of `cache_on_the_fly` and `cache_at_init` can be True!'
            )

        assert isinstance(self.CLASSES, (str, tuple, list))
        if isinstance(self.CLASSES, str):
            self.CLASSES = mmcv.list_from_file(classes)
        if self.PALETTE is None:
            self.PALETTE = self.get_random_palette()

        source_iter = self.get_source_iterator()

        self.samples_list = self.build_samples(
            source_iter, process_fn=self.parse_mat)
        self.num_samples = len(self.samples_list)
        # An error will be raised if failed to load _max_retry_num times in a row
        self._max_retry_num = self.num_samples
        self._retry_count = 0

    def parse_mat(self, source_item):
        img_path, seg_path = source_item
        result = {'filename': img_path, 'seg_filename': seg_path}

        if self.cache_at_init:
            result.update(self.load_image(img_path))
            result.update(self.load_seg_map(seg_path, self.reduce_zero_label))

        return result

    def load_seg_map(self, seg_path, reduce_zero_label):
        gt_semantic_seg = loadmat(seg_path)['S']
        # reduce zero_label
        if reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        return {'gt_semantic_seg': gt_semantic_seg}

    def load_image(self, img_path):
        img = _load_img(img_path, mode='BGR')
        result = {
            'img': img.astype(np.float32),
            'img_shape': img.shape,  # h, w, c
            'ori_shape': img.shape,
        }
        return result

    def build_samples(self, iterable, process_fn):
        samples_list = []
        with Pool(processes=self.num_processes) as p:
            with tqdm(total=len(iterable), desc='Scanning images') as pbar:
                for _, result_dict in enumerate(
                        p.imap_unordered(process_fn, iterable)):
                    if result_dict:
                        samples_list.append(result_dict)
                    pbar.update()

        return samples_list

    def get_source_iterator(self):

        with io.open(self.path, 'r') as f:
            lines = f.read().splitlines()

        img_files = []
        label_files = []
        for line in lines:

            img_filename = os.path.join(self.img_root, line + self.img_suffix)
            label_filename = os.path.join(self.label_root,
                                          line + self.label_suffix)

            if os.path.exists(img_filename) and os.path.exists(label_filename):
                img_files.append(img_filename)
                label_files.append(label_filename)

        return list(zip(img_files, label_files))

    def __getitem__(self, idx):
        result_dict = self.samples_list[idx]
        load_success = True
        try:
            # avoid data cache from taking up too much memory
            if not self.cache_at_init and not self.cache_on_the_fly:
                result_dict = copy.deepcopy(result_dict)

            if not self.cache_at_init:
                if result_dict.get('img', None) is None:
                    result_dict.update(
                        self.load_image(result_dict['filename']))
                if result_dict.get('gt_semantic_seg', None) is None:
                    result_dict.update(
                        self.load_seg_map(
                            result_dict['seg_filename'],
                            reduce_zero_label=self.reduce_zero_label))
                if self.cache_on_the_fly:
                    self.samples_list[idx] = result_dict
            result_dict = self.post_process_fn(copy.deepcopy(result_dict))
            self._retry_count = 0
        except Exception as e:
            logging.warning(e)
            load_success = False

        if not load_success:
            logging.warning(
                'Something wrong with current sample %s,Try load next sample...'
                % result_dict.get('filename', ''))
            self._retry_count += 1
            if self._retry_count >= self._max_retry_num:
                raise ValueError('All samples failed to load!')

            result_dict = self[(idx + 1) % self.num_samples]

        return result_dict


@DATASOURCES.register_module
class SegSourceCocoStuff164k(SegSourceBase):
    CLASSES = [
        'unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
        'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
        'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
        'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
        'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
        'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood'
    ]
    """Data source for semantic segmentation.
    data format is as follows:

    ├── data
    │   │   ├── images
    │   │   │   ├── 1.jpg
    │   │   │   ├── 2.jpg
    │   │   │   ├── ...
    │   │   ├── labels
    │   │   │   ├── 1.png
    │   │   │   ├── 2.png
    │   │   │   ├── ...
    Example1:
        data_source = SegSourceCocoStuff10k(
            label_root='/your/data/labels',
            img_root='/your/data/images',
            classes=${CLASSES}
        )

    Args:
        img_root (str): images dir path
        label_root (str): labels dir path
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
                 img_root,
                 label_root,
                 classes=CLASSES,
                 img_suffix='.jpg',
                 label_suffix='.png',
                 reduce_zero_label=False,
                 palette=None,
                 num_processes=int(cpu_count() / 2),
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 **kwargs) -> None:

        self.img_root = img_root
        self.label_root = label_root

        self.classes = classes
        self.PALETTE = palette
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix

        assert (os.path.exists(self.img_root) and os.path.exists(self.label_root)), \
            f'{self.label_root} or {self.img_root} is not exists'

        super(SegSourceCocoStuff164k, self).__init__(
            classes=classes,
            reduce_zero_label=reduce_zero_label,
            palette=palette,
            parse_fn=parse_raw,
            num_processes=num_processes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)

    def get_source_iterator(self):

        label_files = []
        img_files = []

        label_list = os.listdir(self.label_root)
        for tmp_img in label_list:
            label_file = os.path.join(self.label_root, tmp_img)
            img_file = os.path.join(
                self.img_root,
                tmp_img.replace(self.label_suffix, self.img_suffix))

            if os.path.exists(label_file) and os.path.exists(img_file):

                label_files.append(label_file)
                img_files.append(img_file)

        return list(zip(img_files, label_files))
