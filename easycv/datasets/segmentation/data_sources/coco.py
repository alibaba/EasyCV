# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

from easycv.datasets.registry import DATASOURCES
from easycv.datasets.utils.download_data.download_coco import (
    check_data_exists, download_coco)
from easycv.utils.constant import CACHE_DIR
from .base import load_image


@DATASOURCES.register_module
class SegSourceCoco(object):

    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self,
                 ann_file,
                 img_prefix,
                 palette=None,
                 reduce_zero_label=False,
                 classes=COCO_CLASSES,
                 iscrowd=False) -> None:
        """
        Args:
            ann_file: Path of annotation file.
            img_prefix: coco path prefix
            reduce_zero_label (bool): whether to mark label zero as ignored
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                palette of segmentation map, if none, random palette will be generated
            classes (str | list): classes list or file
            iscrowd: when traing setted as False, when val setted as True
        """

        self.ann_file = ann_file
        self.img_prefile = img_prefix
        self.iscrowd = iscrowd
        self.reduce_zero_label = reduce_zero_label
        if palette is not None:
            self.PALETTE = palette
        else:
            self.PALETTE = self.get_random_palette()

        self.seg = COCO(self.ann_file)
        self.catIds = self.seg.getCatIds(catNms=classes)
        self.imgIds = self._load_annotations(self.seg.getImgIds())

    def _load_annotations(self, imgIds):
        seg_imgIds = []
        for imgId in tqdm(imgIds, desc='Scanning images'):
            annIds = self.seg.getAnnIds(
                imgIds=imgId, catIds=self.catIds, iscrowd=self.iscrowd)
            anns = self.seg.loadAnns(annIds)
            if len(anns):
                seg_imgIds.append(imgId)

        return seg_imgIds

    def load_seg_map(self, gt_semantic_seg, reduce_zero_label):

        # reduce zero_label
        if reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255

        return gt_semantic_seg

    def _parse_load_seg(self, ids):
        annIds = self.seg.getAnnIds(
            imgIds=ids, catIds=self.catIds, iscrowd=self.iscrowd)
        anns = self.seg.loadAnns(annIds)
        pre_cat_mask = self.seg.annToMask(anns[0])
        mask = pre_cat_mask * (self.catIds.index(anns[0]['category_id']) + 1)

        for ann in anns[1:]:

            binary_mask = self.seg.annToMask(ann)
            mask += binary_mask * (self.catIds.index(ann['category_id']) + 1)
            mask_area = pre_cat_mask + binary_mask
            bask_biny = mask_area == 2

            mask[bask_biny] = self.catIds.index(ann['category_id']) + 1
            mask_area[bask_biny] = 1
            pre_cat_mask = mask_area

        return self.load_seg_map(mask, self.reduce_zero_label)

    def get_random_palette(self):
        # Get random state before set seed, and restore
        # random state later.
        # It will prevent loss of randomness, as the palette
        # may be different in each iteration if not specified.
        # See: https://github.com/open-mmlab/mmdetection/issues/5844
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(0, 255, size=(len(self.COCO_CLASSES), 3))
        np.random.set_state(state)

        return palette

    def __len__(self):

        return len(self.imgIds)

    def __getitem__(self, idx):
        imgId = self.imgIds[idx]
        img = self.seg.loadImgs(imgId)[0]
        id = img['id']
        file_name = os.path.join(self.img_prefile, img['file_name'])
        gt_semantic_seg = self._parse_load_seg(id)
        result = {
            'filename': file_name,
            'gt_semantic_seg': gt_semantic_seg,
            'img_fields': ['img'],
            'seg_fields': ['gt_semantic_seg']
        }
        result.update(load_image(file_name))

        return result


@DATASOURCES.register_module
class SegSourceCoco2017(SegSourceCoco):
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(self,
                 download=False,
                 split='train',
                 path=CACHE_DIR,
                 palette=None,
                 reduce_zero_label=False,
                 classes=COCO_CLASSES,
                 iscrowd=False,
                 **kwargs) -> None:
        """
        Args:
            path: This parameter is optional. If download is True and path is not provided,
                    a temporary directory is automatically created for downloading
            download: If the value is True, the file is automatically downloaded to the path directory.
                      If False, automatic download is not supported and data in the path is used
            split: train or val
            reduce_zero_label (bool): whether to mark label zero as ignored
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                palette of segmentation map, if none, random palette will be generated
            classes (str | list): classes list or file
            iscrowd: when traing setted as False, when val setted as True
        """

        if download:
            if path:
                assert os.path.isdir(path), f'{path} is not dir'
                path = download_coco(
                    'coco2017', split=split, target_dir=path, task='detection')
            else:
                path = download_coco('coco2017', split=split, task='detection')
        else:
            if path:
                assert os.path.isdir(path), f'{path} is not dir'
                path = check_data_exists(
                    target_dir=path, split=split, task='detection')
            else:
                raise KeyError('your path is None')
        super().__init__(
            path['ann_file'],
            path['img_prefix'],
            palette=palette,
            reduce_zero_label=reduce_zero_label,
            classes=classes,
            iscrowd=iscrowd)
