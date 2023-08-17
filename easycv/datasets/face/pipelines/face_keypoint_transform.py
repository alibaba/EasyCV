# Copyright (c) Alibaba, Inc. and its affiliates.
import random

import cv2
import imgaug
import imgaug.augmenters as iaa
import numpy as np

from easycv.datasets.registry import PIPELINES

DEST_SIZE = 256
BASE_LANDMARK_NUM = 106
ENLARGE_RATIO = 1.1

CONTOUR_PARTS = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27], [6, 26],
                 [7, 25], [8, 24], [9, 23], [10, 22], [11, 21], [12, 20],
                 [13, 19], [14, 18], [15, 17]]
BROW_PARTS = [[33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [38, 50],
              [39, 49], [40, 48], [41, 47]]
EYE_PARTS = [[66, 79], [67, 78], [68, 77], [69, 76], [70, 75], [71, 82],
             [72, 81], [73, 80], [74, 83]]
NOSE_PARTS = [[55, 65], [56, 64], [57, 63], [58, 62], [59, 61]]
MOUSE_PARTS = [[84, 90], [85, 89], [86, 88], [96, 100], [97, 99], [103, 101],
               [95, 91], [94, 92]]
IRIS_PARTS = [[104, 105]]
MATCHED_PARTS = CONTOUR_PARTS + BROW_PARTS + EYE_PARTS + NOSE_PARTS + MOUSE_PARTS + IRIS_PARTS


def normal():
    """
    3-sigma rule
    return: (-1, +1)
    """
    mu, sigma = 0, 1
    while True:
        s = np.random.normal(mu, sigma)
        if s < mu - 3 * sigma or s > mu + 3 * sigma:
            continue
        return s / 3 * sigma


def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    landmark_ = np.asarray([(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                             M[1, 0] * x + M[1, 1] * y + M[1, 2])
                            for (x, y) in landmark])
    return M, landmark_


class OverLayGenerator:

    def __init__(self, shape):
        # 4x4
        h_seg_len = shape[0] // 4
        w_seg_len = shape[1] // 4

        self.overlay = []
        # 2x2 overlay
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                self.overlay.append((i * w_seg_len, j * h_seg_len,
                                     2 * w_seg_len, 2 * h_seg_len))

        # 2x3 overlay
        for i in range(3):
            for j in range(2):
                if i == 1:
                    continue
                self.overlay.append((i * w_seg_len, j * h_seg_len,
                                     2 * w_seg_len, 3 * h_seg_len))
        for i in range(2):
            for j in range(3):
                if j == 1:
                    continue
                self.overlay.append((i * w_seg_len, j * h_seg_len,
                                     3 * w_seg_len, 2 * h_seg_len))

        # 2x4 overlay
        for i in range(3):
            for j in range(1):
                if i == 1:
                    continue
                self.overlay.append((i * w_seg_len, j * h_seg_len,
                                     2 * w_seg_len, 4 * h_seg_len))
        for i in range(1):
            for j in range(3):
                if j == 1:
                    continue
                self.overlay.append((i * w_seg_len, j * h_seg_len,
                                     4 * w_seg_len, 2 * h_seg_len))


class FaceKeypointsDataAugumentation:

    def __init__(self, input_size):
        # option
        self.enable_flip = True
        self.enable_rotate = True
        self.input_size = input_size

        # mask generator
        coarse_salt_and_pepper_iaa = iaa.CoarseSaltAndPepper(
            (0.25, 0.35), size_percent=(0.03125, 0.015625))
        self.mask_generator = coarse_salt_and_pepper_iaa.mask

        # overlay generator
        self.overlay_generator = OverLayGenerator(shape=(256, 256))

        # flip
        self.mirror_map = FaceKeypointsDataAugumentation.compute_mirror_map()

    @staticmethod
    def compute_mirror_map():

        mirror_map = np.array(range(0, BASE_LANDMARK_NUM), np.int32)
        for x, y in MATCHED_PARTS:
            mirror_map[x] = y
            mirror_map[y] = x

        return mirror_map

    def aug_flip(self, img, pts, visibility, pose):
        # pts[:, 0] = self.input_size - pts[:, 0]
        pts[:, 0] = img.shape[1] - pts[:, 0]
        pts = pts[self.mirror_map]
        if visibility is not None:
            visibility = visibility[self.mirror_map]
        img = cv2.flip(img, 1)
        if pose is not None:
            # fix roll&yaw in pose
            pose['roll'] = -pose['roll']
            pose['yaw'] = -pose['yaw']

        return img, pts, visibility, pose

    def aug_rotate(self, img, pts, pose, angle):
        center = [DEST_SIZE // 2, DEST_SIZE // 2]
        if pose is not None:
            # fix roll in pose
            pose['roll'] += angle

        cx, cy = center
        M, pts = rotate(angle, (cx, cy), pts)

        imgT = cv2.warpAffine(img, M, (int(img.shape[1]), int(img.shape[0])))

        x1 = np.min(pts[:, 0])
        x2 = np.max(pts[:, 0])
        y1 = np.min(pts[:, 1])
        y2 = np.max(pts[:, 1])
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        x1 = int(x1 - (ENLARGE_RATIO - 1.0) / 2.0 * w)
        y1 = int(y1 - (ENLARGE_RATIO - 1.0) * h)

        new_w = int(ENLARGE_RATIO * (1 + normal() * 0.25) * w)
        new_h = int(ENLARGE_RATIO * (1 + normal() * 0.25) * h)
        new_x1 = x1 + int(normal() * DEST_SIZE * 0.15)
        new_y1 = y1 + int(normal() * DEST_SIZE * 0.15)
        new_x2 = new_x1 + new_w
        new_y2 = new_y1 + new_h

        new_xy = new_x1, new_y1
        pts = pts - new_xy

        height, width, _ = imgT.shape
        dx = max(0, -new_x1)
        dy = max(0, -new_y1)
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)

        edx = max(0, new_x2 - width)
        edy = max(0, new_y2 - height)
        new_x2 = min(width, new_x2)
        new_y2 = min(height, new_y2)

        imgT = imgT[new_y1:new_y2, new_x1:new_x2]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            imgT = cv2.copyMakeBorder(
                imgT,
                dy,
                edy,
                dx,
                edx,
                cv2.BORDER_CONSTANT,
                value=(103.94, 116.78, 123.68))

        return imgT, pts, pose

    def random_mask(self, img):
        mask = self.mask_generator.draw_samples(size=img.shape)
        mask = np.expand_dims(np.sum(mask, axis=-1) > 0, axis=-1)
        return mask

    def random_overlay(self):
        index = np.random.choice(len(self.overlay_generator.overlay))
        overlay = self.overlay_generator.overlay[index]
        return overlay

    def augment_blur(self, img):
        h, w = img.shape[:2]
        assert h == w
        ssize = int(random.uniform(0.01, 0.5) * h)
        aug_seq = iaa.Sequential([
            iaa.Sometimes(
                1.0,
                iaa.OneOf([
                    iaa.GaussianBlur((3, 15)),
                    iaa.AverageBlur(k=(3, 15)),
                    iaa.MedianBlur(k=(3, 15)),
                    iaa.MotionBlur((5, 25))
                ])),
            iaa.Resize(ssize, interpolation=imgaug.ALL),
            iaa.Sometimes(
                0.6,
                iaa.OneOf([
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
                    iaa.AdditiveLaplaceNoise(
                        loc=0, scale=(0.0, 0.1 * 255), per_channel=0.5),
                    iaa.AdditivePoissonNoise(lam=(0, 30), per_channel=0.5)
                ])),
            iaa.Sometimes(0.8, iaa.JpegCompression(compression=(40, 90))),
            iaa.Resize(h),
        ])

        aug_img = aug_seq.augment_image(img)
        return aug_img

    def augment_color_temperature(self, img):
        aug = iaa.ChangeColorTemperature((1000, 40000))

        aug_img = aug.augment_image(img)
        return aug_img

    def aug_clr_noise_blur(self, img):
        # skin&light
        if np.random.choice((True, False), p=[0.05, 0.95]):
            img_ycrcb_raw = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            skin_factor_list = [0.6, 0.8, 1.0, 1.2, 1.4]
            skin_factor = np.random.choice(skin_factor_list)
            img_ycrcb_raw[:, :, 0:1] = np.clip(
                img_ycrcb_raw[:, :, 0:1].astype(np.float32) * skin_factor, 0,
                255).astype(np.uint8)
            img = cv2.cvtColor(img_ycrcb_raw, cv2.COLOR_YCR_CB2BGR)

        # gauss blur 5%
        if np.random.choice((True, False), p=[0.05, 0.95]):
            sigma = np.random.choice([0.25, 0.50, 0.75])
            gauss_blur_iaa = iaa.GaussianBlur(sigma=sigma)
            img = gauss_blur_iaa(image=img)

        # gauss noise 5%
        if np.random.choice((True, False), p=[0.05, 0.95]):
            scale = np.random.choice([0.01, 0.03, 0.05])
            gauss_noise_iaa = iaa.AdditiveGaussianNoise(scale=scale * 255)
            img = gauss_noise_iaa(image=img)

        # motion blur 5%
        if np.random.choice((True, False), p=[0.05, 0.95]):
            angle = np.random.choice([0, 45, 90, 135, 180, 225, 270, 315])
            motion_blur_iaa = iaa.MotionBlur(k=5, angle=angle)
            img = motion_blur_iaa(image=img)

        # jpeg compress 5%
        if np.random.choice((True, False), p=[0.05, 0.95]):
            jpeg_compress_iaa = iaa.JpegCompression(compression=(10, 50))
            img = jpeg_compress_iaa(image=img)

        # gamma contrast 5%
        if np.random.choice((True, False), p=[0.05, 0.95]):
            gamma_contrast_iaa = iaa.GammaContrast((0.85, 1.15))
            img = gamma_contrast_iaa(image=img)

        # brightness 5%
        if np.random.choice((True, False), p=[0.05, 0.95]):
            brightness_iaa = iaa.MultiplyAndAddToBrightness(
                mul=(0.85, 1.15), add=(-10, 10))
            img = brightness_iaa(image=img)

        return img

    def augment_set(self, img):
        noisy_image = img.copy().astype(np.uint8)
        if np.random.choice((True, False), p=[0.6, 0.4]):
            aug = iaa.ChangeColorTemperature((1000, 40000))
            noisy_image = aug.augment_image(noisy_image)

        if np.random.choice((True, False), p=[0.8, 0.2]):
            aug_seq = iaa.Sequential([
                iaa.Sometimes(0.5, iaa.JpegCompression(compression=(40, 90))),
                iaa.Sometimes(0.5, iaa.MotionBlur((3, 7))),
                iaa.Sometimes(
                    0.5,
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255))),
            ],
                                     random_order=True)
            noisy_image = aug_seq.augment_image(noisy_image)

        sometimes = lambda aug: iaa.Sometimes(0.25, aug)
        seq = iaa.Sequential([
            sometimes(iaa.AverageBlur(k=(2, 5))),
            sometimes(iaa.GammaContrast((0.5, 2.0)))
        ],
                             random_order=True)

        noisy_image = seq(images=noisy_image)
        return noisy_image


@PIPELINES.register_module()
class FaceKeypointNorm:
    """Data augmentation with Norm.
    """

    def __init__(self, input_size=96):
        self.input_size = input_size

    def __call__(self, results):
        """Perform data augmentation with random image flip."""

        # for key in results.get('img', []):
        if 'img' in results.keys():
            image = results['img']
            h, w, c = image.shape
            image = cv2.resize(image, (self.input_size, self.input_size))
            results['img'] = np.array(image)

            # for key in results.get('target_point', []):
            if 'target_point' in results.keys():
                points = results['target_point']
                points[:, 0] = points[:, 0] / w * float(self.input_size)
                points[:, 1] = points[:, 1] / h * float(self.input_size)
                target_point = np.reshape(points,
                                          (points.shape[0] * points.shape[1]))
                results['target_point'] = np.array(target_point, np.float32)
            else:
                results['target_point'] = np.array(np.zeros(212), np.float32)

            # for key in results.get('target_point_mask', []):
            if 'target_point_mask' in results.keys():
                points_mask = results['target_point_mask']
                points_mask = points_mask.astype(np.float32)
                points_mask = np.reshape(
                    points_mask, (points_mask.shape[0] * points_mask.shape[1]))
                results['target_point_mask'] = points_mask.astype(np.float32)
            else:
                results['target_point_mask'] = np.array(
                    np.zeros(212), np.float32)

            # for key in results.get('target_pose', []):
            if 'target_pose' in results.keys():
                pose = results['target_pose']
                pose = np.asarray([pose['pitch'], pose['roll'], pose['yaw']])
                results['target_pose'] = pose.astype(np.float32)
            else:
                results['target_pose'] = np.array(np.zeros(3), np.float32)

            if 'target_pose_mask' not in results.keys():
                results['target_pose_mask'] = np.array(np.zeros(3), np.float32)

        return results


@PIPELINES.register_module()
class FaceKeypointRandomAugmentation:
    """Data augmentation with random  flip.
    """

    def __init__(self, input_size=96):
        self.input_size = input_size

        # Data Augment
        self.data_aug = FaceKeypointsDataAugumentation(self.input_size)

    def __call__(self, results):
        """Perform data augmentation with random image flip."""

        image = results['img']
        points = results['target_point']
        points_mask = results['target_point_mask']
        pose = results['target_pose']
        pose_mask = results['target_pose_mask']
        overlay_image_path = results['overlay_image_path']

        if np.random.choice((True, False), p=[0.2, 0.8]):
            # overlay
            overlay_pos = self.data_aug.random_overlay()
            overlay_img_index = np.random.choice(len(overlay_image_path))
            overlay_img_filepath = overlay_image_path[overlay_img_index]
            overlay_img = cv2.imread(overlay_img_filepath,
                                     cv2.IMREAD_UNCHANGED)

            (x, y, w, h) = overlay_pos
            x1, y1, x2, y2 = x, y, x + w, y + h
            overlay_img = cv2.resize(overlay_img, dsize=(w, h))
            overlay_mask = overlay_img[:, :, 3:4] / 255.0
            image[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :] * (
                1 - overlay_mask) + overlay_img[:, :, 0:3] * overlay_mask
            image = image.astype(np.uint8)

        angle = pose['roll']
        image, points, pose = self.data_aug.aug_rotate(
            image, points, pose, angle)  # counterclockwise rotate angle
        pose['roll'] = angle  # reset roll=angle

        if np.random.choice((True, False)):
            image_transform, points, _, pose = self.data_aug.aug_flip(
                image, points, None, pose)
        else:
            image_transform = image

        image_transform = self.data_aug.aug_clr_noise_blur(image_transform)

        results['img'] = image_transform
        results['target_point'] = points
        results['target_pose'] = pose
        return results
