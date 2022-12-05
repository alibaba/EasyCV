# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/base/kpt_2d_sview_rgb_img_top_down_dataset.py

import logging

from easycv.datasets.registry import DATASOURCES
from easycv.framework.errors import ValueError
from .top_down import PoseTopDownSource

CROWDPOSE_DATASET_INFO = dict(
    dataset_name='Crowd Pose',
    paper_info=dict(
        author=
        'Jiefeng Li, Can Wang, Hao Zhu, Yihuan Mao, Hao-Shu Fang, Cewu Lu',
        title=
        'CrowdPose: Efficient Crowded Scenes Pose Estimation and A New Benchmark',
        year='2018',
        container='Computer Vision and Pattern Recognition',
        homepage='https://arxiv.org/abs/1812.00324'),
    keypoint_info={
        0:
        dict(
            name='left_shoulder',
            id=0,
            color=[51, 153, 255],
            type='upper',
            swap='left_elbow'),
        1:
        dict(
            name='right_shoulder',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_elbow'),
        2:
        dict(
            name='left_elbow',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_wrist'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_wrist'),
        4:
        dict(
            name='left_wrist',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap=''),
        5:
        dict(
            name='right_wrist', id=5, color=[0, 255, 0], type='upper',
            swap=''),
        6:
        dict(
            name='left_hip',
            id=6,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        7:
        dict(
            name='right_hip',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        8:
        dict(
            name='left_knee',
            id=8,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        9:
        dict(
            name='right_knee',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        10:
        dict(
            name='left_ankle',
            id=10,
            color=[255, 128, 0],
            type='lower',
            swap=''),
        11:
        dict(
            name='right_ankle',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        12:
        dict(
            name='head', id=12, color=[255, 128, 0], type='upper',
            swap='neck'),
        13:
        dict(
            name='neck',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap='left_shoulder'),
    },
    skeleton_info={
        0: dict(link=('head', 'neck'), id=0, color=[0, 255, 0]),
        1: dict(link=('neck', 'left_shoulder'), id=1, color=[0, 255, 0]),
        2: dict(link=('neck', 'right_shoulder'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('left_shoulder', 'left_elbow'), id=3, color=[255, 128, 0]),
        4: dict(link=('left_elbow', 'left_wrist'), id=4, color=[51, 153, 255]),
        5: dict(
            link=('right_shoulder', 'right_elbow'), id=5, color=[51, 153,
                                                                 255]),
        6:
        dict(link=('right_elbow', 'right_wrist'), id=6, color=[51, 153, 255]),
        7: dict(link=('neck', 'right_hip'), id=7, color=[51, 153, 255]),
        8: dict(link=('neck', 'left_hip'), id=8, color=[0, 255, 0]),
        9: dict(link=('right_hip', 'right_knee'), id=9, color=[255, 128, 0]),
        10: dict(link=('right_knee', 'right_ankle'), id=10, color=[0, 255, 0]),
        11: dict(link=('left_hip', 'left_knee'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_knee', 'left_ankle'), id=12, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087
    ])


@DATASOURCES.register_module
class PoseTopDownSourceCrowdPose(PoseTopDownSource):
    """
    CrowdPose keypoint indexes::

    0  'left_shoulder',
    1  'right_shoulder',
    2  'left_elbow',
    3  'right_elbow',
    4  'left_wrist',
    5  'right_wrist',
    6  'left_hip',
    7  'right_hip',
    8  'left_knee',
    9  'right_knee',
    10  'left_ankle',
    11  'right_ankle',
    12  'head',
    13  'neck'

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
                 test_mode=False,
                 **kwargs):
        if dataset_info is None:
            logging.info(
                'dataset_info is missing, use default coco dataset info')
            dataset_info = CROWDPOSE_DATASET_INFO

        self.use_gt_bbox = data_cfg.get('use_gt_bbox', True)
        self.bbox_file = data_cfg.get('bbox_file', None)
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            dataset_info=dataset_info,
            test_mode=test_mode)
