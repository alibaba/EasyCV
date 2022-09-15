# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from
# https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/hand/hand_coco_wholebody_dataset.py
import logging
import os.path as osp

import numpy as np

from easycv.datasets.registry import DATASOURCES
from ..top_down import PoseTopDownSource

COCO_WHOLEBODY_HAND_DATASET_INFO = dict(
    dataset_name='coco_wholebody_hand',
    paper_info=dict(
        author='Jin, Sheng and Xu, Lumin and Xu, Jin and '
        'Wang, Can and Liu, Wentao and '
        'Qian, Chen and Ouyang, Wanli and Luo, Ping',
        title='Whole-Body Human Pose Estimation in the Wild',
        container='Proceedings of the European '
        'Conference on Computer Vision (ECCV)',
        year='2020',
        homepage='https://github.com/jin-s13/COCO-WholeBody/',
    ),
    keypoint_info={
        0:
        dict(name='wrist', id=0, color=[255, 255, 255], type='', swap=''),
        1:
        dict(name='thumb1', id=1, color=[255, 128, 0], type='', swap=''),
        2:
        dict(name='thumb2', id=2, color=[255, 128, 0], type='', swap=''),
        3:
        dict(name='thumb3', id=3, color=[255, 128, 0], type='', swap=''),
        4:
        dict(name='thumb4', id=4, color=[255, 128, 0], type='', swap=''),
        5:
        dict(
            name='forefinger1', id=5, color=[255, 153, 255], type='', swap=''),
        6:
        dict(
            name='forefinger2', id=6, color=[255, 153, 255], type='', swap=''),
        7:
        dict(
            name='forefinger3', id=7, color=[255, 153, 255], type='', swap=''),
        8:
        dict(
            name='forefinger4', id=8, color=[255, 153, 255], type='', swap=''),
        9:
        dict(
            name='middle_finger1',
            id=9,
            color=[102, 178, 255],
            type='',
            swap=''),
        10:
        dict(
            name='middle_finger2',
            id=10,
            color=[102, 178, 255],
            type='',
            swap=''),
        11:
        dict(
            name='middle_finger3',
            id=11,
            color=[102, 178, 255],
            type='',
            swap=''),
        12:
        dict(
            name='middle_finger4',
            id=12,
            color=[102, 178, 255],
            type='',
            swap=''),
        13:
        dict(
            name='ring_finger1', id=13, color=[255, 51, 51], type='', swap=''),
        14:
        dict(
            name='ring_finger2', id=14, color=[255, 51, 51], type='', swap=''),
        15:
        dict(
            name='ring_finger3', id=15, color=[255, 51, 51], type='', swap=''),
        16:
        dict(
            name='ring_finger4', id=16, color=[255, 51, 51], type='', swap=''),
        17:
        dict(name='pinky_finger1', id=17, color=[0, 255, 0], type='', swap=''),
        18:
        dict(name='pinky_finger2', id=18, color=[0, 255, 0], type='', swap=''),
        19:
        dict(name='pinky_finger3', id=19, color=[0, 255, 0], type='', swap=''),
        20:
        dict(name='pinky_finger4', id=20, color=[0, 255, 0], type='', swap='')
    },
    skeleton_info={
        0:
        dict(link=('wrist', 'thumb1'), id=0, color=[255, 128, 0]),
        1:
        dict(link=('thumb1', 'thumb2'), id=1, color=[255, 128, 0]),
        2:
        dict(link=('thumb2', 'thumb3'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('thumb3', 'thumb4'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('wrist', 'forefinger1'), id=4, color=[255, 153, 255]),
        5:
        dict(link=('forefinger1', 'forefinger2'), id=5, color=[255, 153, 255]),
        6:
        dict(link=('forefinger2', 'forefinger3'), id=6, color=[255, 153, 255]),
        7:
        dict(link=('forefinger3', 'forefinger4'), id=7, color=[255, 153, 255]),
        8:
        dict(link=('wrist', 'middle_finger1'), id=8, color=[102, 178, 255]),
        9:
        dict(
            link=('middle_finger1', 'middle_finger2'),
            id=9,
            color=[102, 178, 255]),
        10:
        dict(
            link=('middle_finger2', 'middle_finger3'),
            id=10,
            color=[102, 178, 255]),
        11:
        dict(
            link=('middle_finger3', 'middle_finger4'),
            id=11,
            color=[102, 178, 255]),
        12:
        dict(link=('wrist', 'ring_finger1'), id=12, color=[255, 51, 51]),
        13:
        dict(
            link=('ring_finger1', 'ring_finger2'), id=13, color=[255, 51, 51]),
        14:
        dict(
            link=('ring_finger2', 'ring_finger3'), id=14, color=[255, 51, 51]),
        15:
        dict(
            link=('ring_finger3', 'ring_finger4'), id=15, color=[255, 51, 51]),
        16:
        dict(link=('wrist', 'pinky_finger1'), id=16, color=[0, 255, 0]),
        17:
        dict(
            link=('pinky_finger1', 'pinky_finger2'), id=17, color=[0, 255, 0]),
        18:
        dict(
            link=('pinky_finger2', 'pinky_finger3'), id=18, color=[0, 255, 0]),
        19:
        dict(
            link=('pinky_finger3', 'pinky_finger4'), id=19, color=[0, 255, 0])
    },
    joint_weights=[1.] * 21,
    sigmas=[
        0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018,
        0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019, 0.022,
        0.031
    ])


@DATASOURCES.register_module()
class HandCocoPoseTopDownSource(PoseTopDownSource):
    """Coco Whole-Body-Hand Source for top-down hand pose estimation.

        "Whole-Body Human Pose Estimation in the Wild", ECCV'2020.
        More details can be found in the `paper
        <https://arxiv.org/abs/2007.11858>`__ .

        The dataset loads raw features and apply specified transforms
        to return a dict containing the image tensors and other information.

        COCO-WholeBody Hand keypoint indexes::

            0: 'wrist',
            1: 'thumb1',
            2: 'thumb2',
            3: 'thumb3',
            4: 'thumb4',
            5: 'forefinger1',
            6: 'forefinger2',
            7: 'forefinger3',
            8: 'forefinger4',
            9: 'middle_finger1',
            10: 'middle_finger2',
            11: 'middle_finger3',
            12: 'middle_finger4',
            13: 'ring_finger1',
            14: 'ring_finger2',
            15: 'ring_finger3',
            16: 'ring_finger4',
            17: 'pinky_finger1',
            18: 'pinky_finger2',
            19: 'pinky_finger3',
            20: 'pinky_finger4'

        Args:
            ann_file (str): Path to the annotation file.
            img_prefix (str): Path to a directory where images are held.
                Default: None.
            data_cfg (dict): config
            dataset_info (DatasetInfo): A class containing all dataset info.
            test_mode (bool): Store True when building test or
                validation dataset. Default: False.
        """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            logging.info(
                'dataset_info is missing, use default coco wholebody hand dataset info'
            )
            dataset_info = COCO_WHOLEBODY_HAND_DATASET_INFO

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                for type in ['left', 'right']:
                    if obj[f'{type}hand_valid'] and max(
                            obj[f'{type}hand_kpts']) > 0:
                        joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                        joints_3d_visible = np.zeros((num_joints, 3),
                                                     dtype=np.float32)

                        keypoints = np.array(obj[f'{type}hand_kpts']).reshape(
                            -1, 3)
                        joints_3d[:, :2] = keypoints[:, :2]
                        joints_3d_visible[:, :2] = np.minimum(
                            1, keypoints[:, 2:3])

                        image_file = osp.join(self.img_prefix,
                                              self.id2name[img_id])
                        center, scale = self._xywh2cs(
                            *obj[f'{type}hand_box'][:4])
                        gt_db.append({
                            'image_file': image_file,
                            'image_id': img_id,
                            'rotation': 0,
                            'center': center,
                            'scale': scale,
                            'joints_3d': joints_3d,
                            'joints_3d_visible': joints_3d_visible,
                            'dataset': self.dataset_name,
                            'bbox': obj[f'{type}hand_box'],
                            'bbox_score': 1,
                            'bbox_id': bbox_id
                        })
                        bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db
