# Copyright (c) Alibaba, Inc. and its affiliates.
import random

import cv2
import numpy as np

from easycv.datasets.registry import PIPELINES


@PIPELINES.register_module()
class SegRandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class ColorAugSSDTransform(object):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()
        self.brightness_delta = brightness_delta
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high
        self.saturation_low = saturation_low
        self.saturation_high = saturation_high
        self.hue_delta = hue_delta
        assert img_format in ['BGR', 'RGB']
        self.is_rgb = img_format == 'RGB'
        del img_format

    # def apply_coords(self, coords):
    #     return coords

    # def apply_segmentation(self, segmentation):
    #     return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        img = self.brightness(img)
        if random.randrange(2):
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randrange(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        if random.randrange(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_low,
                                     self.saturation_high))
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, results):
        img = results['img']
        img = self.apply_image(img)
        results['img'] = img
        return results
