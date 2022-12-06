# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Tuple

import mmcv
import numpy as np
from numpy import random

from easycv.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                              LiDARInstance3DBoxes)
from easycv.datasets.registry import PIPELINES
from .functional import (horizontal_flip_bbox, horizontal_flip_cam_params,
                         horizontal_flip_canbus,
                         horizontal_flip_image_multiview,
                         scale_image_multiple_view)


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                              self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class ObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class ObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results['img']
        ]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results['img']
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val)
                for img in results['img']
            ]

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Resize the multiple-view images with the same scale selected randomly.  .
    Args:
        scales (tuple of float): ratio for resizing the images. Every time, select one ratio
        randomly.
    """

    def __init__(self, scales=[0.5, 1.0, 1.5]):
        self.scales = scales
        self.seed = 0

    def forward(
        self,
        imgs: List[np.ndarray],
        cam_intrinsics: List[np.ndarray],
        lidar2img: List[np.ndarray],
        seed=None,
        scale=1
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Args:
            imgs (list of numpy.array): Multiple-view images to be resized. len(img) is the number of cameras.
                    img shape: [H, W, 3].
        cam_intrinsics (list of numpy.array): Intrinsic parameters of different cameras. Transformations from camera
                    to image. len(cam_intrinsics) is the number of camera. For each camera, shape is 4 * 4.
        cam_extrinsics (list of numpy.array): Extrinsic parameters of different cameras. Transformations from
                lidar to cameras. len(cam_extrinsics) is the number of camera. For each camera, shape is 4 * 4.
        lidar2img (list of numpy.array): Transformations from lidar to images. len(lidar2img) is the number
                of camera. For each camera, shape is 4 * 4.
            seed (int): Seed for generating random number.
        Returns:
            imgs_new (list of numpy.array): Updated multiple-view images
            cam_intrinsics_new (list of numpy.array): Updated intrinsic parameters of different cameras.
            lidar2img_new (list of numpy.array): Updated Transformations from lidar to images.
        """
        rand_scale = scale
        imgs_new, cam_intrinsic_new, lidar2img_new = scale_image_multiple_view(
            imgs, cam_intrinsics, lidar2img, rand_scale)

        return imgs_new, cam_intrinsic_new, lidar2img_new

    def __call__(self, data):
        imgs = data['img']
        cam_intrinsics = data['cam_intrinsic']
        lidar2img = data['lidar2img']

        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        scale = data[
            'resize_scale'] if 'resize_scale' in data else self.scales[rand_ind]

        imgs_new, cam_intrinsic_new, lidar2img_new = self.forward(
            imgs, cam_intrinsics, lidar2img, None, scale)

        data['img'] = imgs_new
        data['cam_intrinsic'] = cam_intrinsic_new
        data['lidar2img'] = lidar2img_new
        return data


@PIPELINES.register_module()
class RandomHorizontalFlipMultiViewImage(object):
    """Horizontally flip the multiple-view images with bounding boxes, camera parameters and can bus randomly.  .
    Support coordinate systems like Waymo (https://waymo.com/open/data/perception/) or Nuscenes (https://www.nuscenes.org/public/images/data.png).
    Args:
        flip_ratio (float 0~1): probability of the images being flipped. Default value is 0.5.
        dataset (string): Specify 'waymo' coordinate system or 'nuscenes' coordinate system.
    """

    def __init__(self, flip_ratio=0.5, dataset='nuScenes'):
        self.flip_ratio = flip_ratio
        self.seed = 0
        self.dataset = dataset

    def forward(
        self,
        imgs: List[np.ndarray],
        bboxes_3d: np.ndarray,
        cam_intrinsics: List[np.ndarray],
        cam_extrinsics: List[np.ndarray],
        lidar2imgs: List[np.ndarray],
        canbus: np.ndarray,
        seed=None,
        flip_flag=True
    ) -> Tuple[bool, List[np.ndarray], np.ndarray, List[np.ndarray],
               List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Args:
        imgs (list of numpy.array): Multiple-view images to be resized. len(img) is the number of cameras.
                img shape: [H, W, 3].
        bboxes_3d (np.ndarray): bounding boxes of shape [N * 7], N is the number of objects.
        cam_intrinsics (list of numpy.array): Intrinsic parameters of different cameras. Transformations from camera
                to image. len(cam_intrinsics) is the number of camera. For each camera, shape is 4 * 4.
        cam_extrinsics (list of numpy.array): Extrinsic parameters of different cameras. Transformations from
                lidar to cameras. len(cam_extrinsics) is the number of camera. For each camera, shape is 4 * 4.
        lidar2img (list of numpy.array): Transformations from lidar to images. len(lidar2img) is the number
                of camera. For each camera, shape is 4 * 4.
        canbus (numpy.array):
        seed (int): Seed for generating random number.
        Returns:
            imgs_new (list of numpy.array): Updated multiple-view images
            cam_intrinsics_new (list of numpy.array): Updated intrinsic parameters of different cameras.
            lidar2img_new (list of numpy.array): Updated Transformations from lidar to images.
        """

        if flip_flag == False:
            return flip_flag, imgs, bboxes_3d, cam_intrinsics, cam_extrinsics, lidar2imgs, canbus
        else:
            # flip_flag = True
            imgs_flip = horizontal_flip_image_multiview(imgs)
            bboxes_3d_flip = horizontal_flip_bbox(bboxes_3d, self.dataset)
            img_shape = imgs[0].shape
            cam_intrinsics_flip, cam_extrinsics_flip, lidar2imgs_flip = horizontal_flip_cam_params(
                img_shape, cam_intrinsics, cam_extrinsics, lidar2imgs,
                self.dataset)
            canbus_flip = horizontal_flip_canbus(canbus, self.dataset)
        return flip_flag, imgs_flip, bboxes_3d_flip, cam_intrinsics_flip, cam_extrinsics_flip, lidar2imgs_flip, canbus_flip

    def __call__(self, data):

        imgs = data['img']
        bboxes_3d = data['gt_bboxes_3d']
        cam_intrinsics = data['cam_intrinsic']
        lidar2imgs = data['lidar2img']
        canbus = data['can_bus']
        cam_extrinsics = data['lidar2cam']
        flip_flag = data['flip_flag']

        flip_flag, imgs_flip, bboxes_3d_flip, cam_intrinsics_flip, cam_extrinsics_flip, lidar2imgs_flip, canbus_flip = self.forward(
            imgs, bboxes_3d, cam_intrinsics, cam_extrinsics, lidar2imgs,
            canbus, None, flip_flag)

        data['img'] = imgs_flip
        data['gt_bboxes_3d'] = bboxes_3d_flip
        data['cam_intrinsic'] = cam_intrinsics_flip
        data['lidar2img'] = lidar2imgs_flip
        data['can_bus'] = canbus_flip
        data['lidar2cam'] = cam_extrinsics_flip
        return data
