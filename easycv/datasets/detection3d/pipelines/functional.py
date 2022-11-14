######################################################################
# Copyright (c) 2022 OpenPerceptionX. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
######################################################################

######################################################################
# This file includes concrete implementation for different data augmentation
# methods in transforms.py.
######################################################################

from typing import List, Tuple

import cv2
import numpy as np

#  Available interpolation modes (opencv)
cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def scale_image_multiple_view(
    imgs: List[np.ndarray],
    cam_intrinsics: List[np.ndarray],
    #   cam_extrinsics: List[np.ndarray],
    lidar2img: List[np.ndarray],
    rand_scale: float,
    interpolation='bilinear'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Resize the multiple-view images with the same scale selected randomly.
    Notably used in :class:`.transforms.RandomScaleImageMultiViewImage_naive
    Args:
        imgs (list of numpy.array): Multiple-view images to be resized. len(img) is the number of cameras.
                    img shape: [H, W, 3].
        cam_intrinsics (list of numpy.array): Intrinsic parameters of different cameras. Transformations from camera
                    to image. len(cam_intrinsics) is the number of camera. For each camera, shape is 4 * 4.
        cam_extrinsics (list of numpy.array): Extrinsic parameters of different cameras. Transformations from
                lidar to cameras. len(cam_extrinsics) is the number of camera. For each camera, shape is 4 * 4.
        lidar2img (list of numpy.array): Transformations from lidar to images. len(lidar2img) is the number
                of camera. For each camera, shape is 4 * 4.
        rand_scale (float): resize ratio
        interpolation (string): mode for interpolation in opencv.
    Returns:
        imgs_new (list of numpy.array): Updated multiple-view images
        cam_intrinsics_new (list of numpy.array): Updated intrinsic parameters of different cameras.
        lidar2img_new (list of numpy.array): Updated Transformations from lidar to images.
    """
    y_size = [int(img.shape[0] * rand_scale) for img in imgs]
    x_size = [int(img.shape[1] * rand_scale) for img in imgs]
    scale_factor = np.eye(4)
    scale_factor[0, 0] *= rand_scale
    scale_factor[1, 1] *= rand_scale
    imgs_new = [
        cv2.resize(
            img, (x_size[idx], y_size[idx]),
            interpolation=cv2_interp_codes[interpolation])
        for idx, img in enumerate(imgs)
    ]
    cam_intrinsics_new = [
        scale_factor @ cam_intrinsic for cam_intrinsic in cam_intrinsics
    ]
    lidar2img_new = [scale_factor @ l2i for l2i in lidar2img]

    return imgs_new, cam_intrinsics_new, lidar2img_new


def horizontal_flip_image_multiview(
        imgs: List[np.ndarray]) -> List[np.ndarray]:
    """Flip every image horizontally.
    Args:
        imgs (list of numpy.array): Multiple-view images to be resized. len(img) is the number of cameras.
                    img shape: [H, W, 3].
    Returns:
        imgs_new (list of numpy.array): Flippd multiple-view images
    """
    imgs_new = [np.flip(img, axis=1) for img in imgs]
    return imgs_new


def vertical_flip_image_multiview(imgs: List[np.ndarray]) -> List[np.ndarray]:
    """Flip every image vertically.
    Args:
        imgs (list of numpy.array): Multiple-view images to be resized. len(img) is the number of cameras.
                    img shape: [H, W, 3].
    Returns:
        imgs_new (list of numpy.array): Flippd multiple-view images
    """
    imgs_new = [np.flip(img, axis=0) for img in imgs]
    return imgs_new


def horizontal_flip_bbox(bboxes_3d: np.ndarray, dataset: str) -> np.ndarray:
    """Flip bounding boxes horizontally.
    Args:
        bboxes_3d (np.ndarray): bounding boxes of shape [N * 7], N is the number of objects.
        dataset (string): 'waymo' coordinate system or 'nuscenes' coordinate system.
    Returns:
        bboxes_3d (numpy.array): Flippd bounding boxes.
    """
    if dataset == 'nuScenes':
        bboxes_3d.tensor[:, 0::7] = -bboxes_3d.tensor[:, 0::7]
        bboxes_3d.tensor[:, 6] = -bboxes_3d.tensor[:, 6]  # + np.pi
    elif dataset == 'waymo':
        bboxes_3d[:, 1::7] = -bboxes_3d[:, 1::7]
        bboxes_3d[:, 6] = -bboxes_3d[:, 6] + np.pi
    return bboxes_3d


def horizontal_flip_cam_params(
    img_shape: np.ndarray, cam_intrinsics: List[np.ndarray],
    cam_extrinsics: List[np.ndarray], lidar2imgs: List[np.ndarray],
    dataset: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Flip camera parameters horizontally.
    Args:
        img_shape (numpy.array) of shape [3].
        cam_intrinsics (list of numpy.array): Intrinsic parameters of different cameras. Transformations from camera
                    to image. len(cam_intrinsics) is the number of camera. For each camera, shape is 4 * 4.
        cam_extrinsics (list of numpy.array): Extrinsic parameters of different cameras. Transformations from
                lidar to cameras. len(cam_extrinsics) is the number of camera. For each camera, shape is 4 * 4.
        lidar2img (list of numpy.array): Transformations from lidar to images. len(lidar2img) is the number
                of camera. For each camera, shape is 4 * 4.
        dataset (string): Specify 'waymo' coordinate system or 'nuscenes' coordinate system.
    Returns:
        cam_intrinsics (list of numpy.array): Updated intrinsic parameters of different cameras.
        cam_extrinsics (list of numpy.array): Updated extrinsic parameters of different cameras.
        lidar2img (list of numpy.array): Updated Transformations from lidar to images.
    """
    flip_factor = np.eye(4)
    lidar2imgs = []

    w = img_shape[1]
    if dataset == 'nuScenes':
        flip_factor[0, 0] = -1
        cam_extrinsics = [l2c @ flip_factor for l2c in cam_extrinsics]
        for cam_intrinsic, l2c in zip(cam_intrinsics, cam_extrinsics):
            cam_intrinsic[0, 0] = -cam_intrinsic[0, 0]
            cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
            lidar2imgs.append(cam_intrinsic @ l2c)
    elif dataset == 'waymo':
        flip_factor[1, 1] = -1
        cam_extrinsics = [l2c @ flip_factor for l2c in cam_extrinsics]
        for cam_intrinsic, l2c in zip(cam_intrinsics, cam_extrinsics):
            cam_intrinsic[0, 0] = -cam_intrinsic[0, 0]
            cam_intrinsic[0, 2] = w - cam_intrinsic[0, 2]
            lidar2imgs.append(cam_intrinsic @ l2c)
    else:
        assert False

    return cam_intrinsics, cam_extrinsics, lidar2imgs


def horizontal_flip_canbus(canbus: np.ndarray, dataset: str) -> np.ndarray:
    """Flip can bus horizontally.
    Args:
        canbus (numpy.ndarray) of shape [18,]
        dataset (string): 'waymo' or 'nuscenes'
    Returns:
        canbus_new (list of numpy.array): Flipped canbus.
    """
    if dataset == 'nuScenes':
        # results['canbus'][1] = -results['canbus'][1]  # flip location
        # results['canbus'][-2] = -results['canbus'][-2]  # flip direction
        canbus[-1] = -canbus[-1]  # flip direction
    elif dataset == 'waymo':
        # results['canbus'][1] = -results['canbus'][-1]  # flip location
        # results['canbus'][-2] = -results['canbus'][-2]  # flip direction
        canbus[-1] = -canbus[-1]  # flip direction
    else:
        raise NotImplementedError((f'Not support {dataset} dataset'))
    return canbus
