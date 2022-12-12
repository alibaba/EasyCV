import copy
import json
import logging
import os

import cv2
import numpy as np
import torch

from easycv.datasets.face.pipelines.face_keypoint_transform import (
    FaceKeypointNorm, FaceKeypointRandomAugmentation, normal)
from easycv.datasets.registry import DATASOURCES
from easycv.file.image import load_image

FACE_KEYPOINT_DATASET_INFO = dict(
    real_list_file_dir='real_face_list.txt',
    data_info_dir='infos/merge/',
    data_image_dir='images/merge/',
    data_overlay_dir='images/overlay/',
)


@DATASOURCES.register_module()
class FaceKeypintSource():
    """
        load dataset for face key points
    """

    def __init__(self,
                 data_cfg,
                 data_range,
                 real_list_path=None,
                 info_path=None,
                 image_path=None,
                 data_overlay_path=None,
                 dataset_info=None,
                 **kwargs):
        super(FaceKeypintSource, self).__init__()
        """
        Args:
            data_cfg: Data config dict
            data_range: rang of dataset for training or validation
            real_list_file_path: path of file contains image list
            data_info_dir: annotation file path
            data_img_dir: image file path
            data_overlay_dir: overlay background image path

            dataset_info: A dict containing all dataset info
        """
        if dataset_info is None:
            logging.info(
                'dataset_info is missing, use default face keypoiny dataset info'
            )
            dataset_info = FACE_KEYPOINT_DATASET_INFO

        data_root = data_cfg['data_root']
        real_list_file_path = os.path.join(data_root,
                                           dataset_info['real_list_file_dir'])
        data_info_dir = os.path.join(data_root, dataset_info['data_info_dir'])
        data_img_dir = os.path.join(data_root, dataset_info['data_image_dir'])
        data_overlay_dir = os.path.join(data_root,
                                        dataset_info['data_overlay_dir'])
        self.input_size = data_cfg['input_size']
        data_range = data_range

        if real_list_path is not None:
            real_list_file_path = real_list_path
        if info_path is not None:
            data_info_dir = info_path
        if image_path is not None:
            data_img_dir = image_path
        if data_overlay_path is not None:
            data_overlay_dir = data_overlay_path

        # overlay
        self.overlay_image_path = []
        for overlay_img_file in sorted(os.listdir(data_overlay_dir)):
            overlay_img_filepath = os.path.join(data_overlay_dir,
                                                overlay_img_file)
            self.overlay_image_path.append(overlay_img_filepath)

        self.points_and_pose_datas = []
        with open(real_list_file_path, 'r') as real_list_file:
            real_list_lines = real_list_file.readlines()
        for index in range(data_range[0], data_range[1]):
            idx = int(real_list_lines[index])
            img_path = os.path.join(data_img_dir, '{:06d}.png'.format(idx))
            if not os.path.exists(img_path):
                logging.warning('image %s does not exist' % img_path)
                continue
            info_path = os.path.join(data_info_dir, '{:06d}.json'.format(idx))
            if not os.path.exists(info_path):
                logging.warning('annotation %s does not exist' % info_path)
                continue
            with open(info_path, 'r') as info_file:
                info_json = json.load(info_file)
                assert info_json['face_count'] == 1
                base_info = info_json['face_infos'][0]['base_info']

                # points
                assert base_info['points_array'] is not None
                points = np.asarray(base_info['points_array']).astype(
                    np.float32)
                points_mask = np.abs(points - (-999)) > 0.0001

                # pose
                pose = {'pitch': -999, 'yaw': -999, 'roll': -999}
                if base_info['pitch'] is not None and base_info[
                        'yaw'] is not None and base_info['roll'] is not None:
                    pose['pitch'] = base_info['pitch']
                    pose['yaw'] = base_info['yaw']
                    # pose["roll"] = base_info["roll"]
                    # datasets have been preprocessed, roll=0
                    # add noise to pose
                    pose['roll'] = normal() * 10.0

                pose_mask = np.asarray([
                    np.abs(pose['pitch'] - (-999)) > 0.0001,
                    np.abs(pose['roll'] - (-999)) > 0.0001,
                    np.abs(pose['yaw'] - (-999)) > 0.0001
                ])

            self.points_and_pose_datas.append(
                (img_path, points, points_mask, pose, pose_mask))

        self.db = []
        for img_path, points, points_mask, pose, pose_mask in copy.deepcopy(
                self.points_and_pose_datas):
            image = cv2.imread(img_path)

            points[:,
                   0] = points[:, 0] / image.shape[1] * float(self.input_size)
            points[:,
                   1] = points[:, 1] / image.shape[0] * float(self.input_size)

            target_point = np.reshape(points,
                                      (points.shape[0] * points.shape[1]))
            points_mask = points_mask.astype(np.float32)
            points_mask = np.reshape(
                points_mask, (points_mask.shape[0] * points_mask.shape[1]))
            pose = np.asarray([pose['pitch'], pose['roll'], pose['yaw']])

            self.db.append({
                'img_path':
                img_path,
                'target_point':
                torch.tensor(np.array(target_point, np.float32)),
                'target_point_mask':
                torch.tensor(points_mask),
                'target_pose':
                torch.tensor(np.array(pose, np.float32)),
                'target_pose_mask':
                torch.tensor(pose_mask.astype(np.float32))
            })

    def __getitem__(self, index):
        img_path, points, points_mask, pose, pose_mask = copy.deepcopy(
            self.points_and_pose_datas[index])

        image = load_image(img_path, backend='cv2')

        return {
            'img': image,
            'target_point': points,
            'target_point_mask': points_mask,
            'target_pose': pose,
            'target_pose_mask': pose_mask,
            'overlay_image_path': self.overlay_image_path
        }

    def __len__(self):
        return len(self.points_and_pose_datas)
