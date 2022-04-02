# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from mmcv.runner.dist_utils import get_dist_info
from PIL import Image
from tqdm import tqdm

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from easycv.utils.constant import MAX_READ_IMAGE_TRY_TIMES

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']


def parse_xml(xml_path, classes):
    with io.open(xml_path, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        gt_bboxes = []
        gt_labels = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if int(difficult) == 1:
                continue
            if cls not in classes:
                logging.warning(
                    'class: %s not in given class list, skip the object!' %
                    cls)
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            box = (float(xmlbox.find('xmin').text),
                   float(xmlbox.find('ymin').text),
                   float(xmlbox.find('xmax').text),
                   float(xmlbox.find('ymax').text))
            gt_bboxes.append(box)
            gt_labels.append(cls_id)

    if len(gt_bboxes) == 0:
        gt_bboxes = np.zeros((0, 5), dtype=np.float32)

    img_info = {
        'gt_bboxes': np.array(gt_bboxes, dtype=np.float32),
        'gt_labels': np.array(gt_labels, dtype=np.int64)
    }

    return img_info


@DATASOURCES.register_module
class DetSourceVOC(object):
    """
    data dir is as follows:
    ```
    |- voc_data
        |-ImageSets
            |-Main
                |-train.txt
                |-...
        |-JPEGImages
            |-00001.jpg
            |-...
        |-Annotations
            |-00001.xml
            |-...

    ```
    Example1:
        data_source = DetSourceVOC(
            path='/your/voc_data/ImageSets/Main/train.txt',
            classes=${VOC_CLASSES},
        )
    Example1:
        data_source = DetSourceVOC(
            path='/your/voc_data/train.txt',
            classes=${VOC_CLASSES},
            img_root_path='/your/voc_data/images',
            img_root_path='/your/voc_data/annotations'
        )
    """

    def __init__(self,
                 path,
                 classes=[],
                 img_root_path=None,
                 label_root_path=None,
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 img_suffix='.jpg',
                 label_suffix='.xml',
                 **kwargs):
        """
        Args:
            path: path of img id list file in ImageSets/Main/
            classes: classes list
            img_root_path: image dir path, if None, default to detect the image dir by the relative path of the `path`
                according to the VOC data format.
            label_root_path: label dir path, if None, default to detect the label dir by the relative path of the `path`
                according to the VOC data format.
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            img_suffix: suffix of image file
            label_suffix: suffix of label file
        """
        self.CLASSES = classes
        self.rank, self.world_size = get_dist_info()
        self.path = path
        self.img_root_path = img_root_path
        self.label_root_path = label_root_path
        self.cache_at_init = cache_at_init
        self.cache_on_the_fly = cache_on_the_fly

        if not img_root_path:
            self.img_root_path = os.path.join(
                self.path.split('ImageSets/Main')[0], 'JPEGImages')
        if not self.label_root_path:
            self.label_root_path = os.path.join(
                self.path.split('ImageSets/Main')[0], 'Annotations')

        imgs_path_list = []
        labels_path_list = []
        with io.open(self.path, 'r') as t:
            id_lines = t.read().splitlines()
            for id_line in id_lines:
                img_id = id_line.strip().split(' ')[0]
                img_path = os.path.join(self.img_root_path,
                                        img_id + img_suffix)
                imgs_path_list.append(img_path)

                label_path = os.path.join(self.label_root_path,
                                          img_id + label_suffix)
                labels_path_list.append(label_path)

        # TODO: filter bad sample
        self.samples_list = self.build_samples(
            list(zip(imgs_path_list, labels_path_list)))

    def get_source_info(self, img_and_label):
        img_path = img_and_label[0]
        label_path = img_and_label[1]
        source_info = parse_xml(label_path, self.CLASSES)
        source_info.update({'filename': img_path})

        return source_info

    def _build_sample_from_source_info(self, source_info):
        if 'filename' not in source_info:
            return {}

        result_dict = source_info

        img_info = self.load_image(source_info['filename'])
        result_dict.update(img_info)

        result_dict.update({
            'img_fields': ['img'],
            'bbox_fields': ['gt_bboxes']
        })

        return result_dict

    def build_sample(self, data):
        result_dict = self.get_source_info(data)

        if self.cache_at_init:
            result_dict = self._build_sample_from_source_info(result_dict)

        return result_dict

    def build_samples(self, iterable):
        samples_list = []
        proc_num = int(cpu_count() / 2)
        with Pool(processes=proc_num) as p:
            with tqdm(total=len(iterable), desc='Scanning images') as pbar:
                for _, result_dict in enumerate(
                        p.imap_unordered(self.build_sample, iterable)):
                    if result_dict:
                        samples_list.append(result_dict)
                    pbar.update()

        return samples_list

    def load_image(self, img_path):
        result = {}
        try_cnt = 0
        img = None
        while try_cnt < MAX_READ_IMAGE_TRY_TIMES:
            try:
                with io.open(img_path, 'rb') as infile:
                    # cv2.imdecode may corrupt when the img is broken
                    image = Image.open(infile)
                    img = cv2.cvtColor(
                        np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                    assert img is not None, 'Image load error, try %s : %s' % (
                        try_cnt, img_path)
                    break
            except:
                time.sleep(2)
            try_cnt += 1

        if img is None:
            raise ValueError('Read Image Times Out: ' + img_path)

        result['img'] = img.astype(np.float32)
        result['img_shape'] = img.shape  # h, w, c
        result['ori_img_shape'] = img.shape

        return result

    def get_length(self):
        return len(self.samples_list)

    def __len__(self):
        return self.get_length()

    def get_ann_info(self, idx):
        """
        Get raw annotation info, include bounding boxes, labels and so on.
        `bboxes` format is as [x1, y1, x2, y2] without normalization.
        """
        sample_info = self.samples_list[idx]
        if sample_info.get('gt_labels', None) is None:
            sample_info = self._build_sample_from_source_info(sample_info)
            if self.cache_at_init or self.cache_on_the_fly:
                self.samples_list[idx] = sample_info

        annotations = {
            'bboxes': sample_info['gt_bboxes'],
            'labels': sample_info['gt_labels'],
            'groundtruth_is_crowd': np.zeros_like(sample_info['gt_labels'])
        }

        return annotations

    def get_sample(self, idx):
        result_dict = self.samples_list[idx]
        try:
            if result_dict.get('img', None) is None:
                result_dict = self._build_sample_from_source_info(result_dict)
                if self.cache_at_init or self.cache_on_the_fly:
                    self.samples_list[idx] = result_dict
        except Exception as e:
            logging.warning(e)

        if not result_dict:
            logging.warning(
                'Something wrong with current sample %s,Try load next sample...'
                % result_dict.get('filename', ''))
            result_dict = self.get_sample(idx + 1)

        return result_dict
