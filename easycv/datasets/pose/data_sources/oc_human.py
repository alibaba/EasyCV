# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/top_down/topdown_coco_dataset.py
import json
import logging
import os

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.framework.errors import ValueError
from .top_down import PoseTopDownSource

OC_HUMAN_DATASET_INFO = dict(
    dataset_name='OC HUMAN',
    paper_info=dict(
        author=
        'Song-Hai Zhang, Ruilong Li, Xin Dong, Paul L. Rosin, Zixi Cai, Han Xi, Dingcheng Yang, Hao-Zhi Huang, Shi-Min Hu',
        title='Pose2Seg: Detection Free Human Instance Segmentation',
        container='Computer Vision and Pattern Recognition',
        year='2019',
        homepage='https://github.com/liruilong940607/OCHumanApi'),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1:
        dict(
            name='left_eye',
            id=1,
            color=[51, 153, 255],
            type='upper',
            swap='right_eye'),
        2:
        dict(
            name='right_eye',
            id=2,
            color=[51, 153, 255],
            type='upper',
            swap='left_eye'),
        3:
        dict(
            name='left_ear',
            id=3,
            color=[51, 153, 255],
            type='upper',
            swap='right_ear'),
        4:
        dict(
            name='right_ear',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_ear'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='right_shoulder',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        7:
        dict(
            name='left_elbow',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        8:
        dict(
            name='right_elbow',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        9:
        dict(
            name='left_wrist',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        10:
        dict(
            name='right_wrist',
            id=10,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        11:
        dict(
            name='left_hip',
            id=11,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        12:
        dict(
            name='right_hip',
            id=12,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='right_knee',
            id=14,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        15:
        dict(
            name='left_ankle',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        16:
        dict(
            name='right_ankle',
            id=16,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle')
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7:
        dict(
            link=('left_shoulder', 'right_shoulder'),
            id=7,
            color=[51, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13:
        dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16:
        dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17:
        dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18:
        dict(
            link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])


@DATASOURCES.register_module
class PoseTopDownSourceChHuman(PoseTopDownSource):
    """Oc Human Source for top-down pose estimation.

    `Pose2Seg: Detection Free Human Instance Segmentation' ECCV'2019
    More details can be found in the `paper
    <https://arxiv.org/abs/1803.10683>`__ .

    The source loads raw features to build a data meta object
    containing the image info, annotation info and others.

    Oc Human keypoint indexes::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        subset: Applicable to non-coco or coco style data sets,
                if subset == train or val or test, in non-coco style
                else subset == None , in coco style
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or

    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 subset=None,
                 dataset_info=None,
                 test_mode=False,
                 **kwargs):

        if dataset_info is None:
            logging.info(
                'dataset_info is missing, use default coco dataset info')
            dataset_info = OC_HUMAN_DATASET_INFO

        self.subset = subset

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            coco_style=not bool(subset),  # bool(1-bool(subset))
            dataset_info=dataset_info,
            test_mode=test_mode)

    def _get_db(self):
        """Load dataset."""
        # ground truth bbox
        if self.subset:
            gt_db = self._load_keypoint_annotations()
        else:
            gt_db = super()._load_keypoint_annotations()

        return gt_db

    def _load_keypoint_annotations(self):
        self._load_annofile()
        gt_db = list()
        for img_id in self.imgIds:
            gt_db.extend(self._oc_load_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_annofile(self):
        self.human = json.load(open(self.ann_file, 'r'))

        self.keypoint_names = self.human['keypoint_names']
        self.keypoint_visible = self.human['keypoint_visible']

        self.images = {}
        self.imgIds = []
        for imgItem in self.human['images']:
            annos = [
                anno for anno in imgItem['annotations'] if anno['keypoints']
            ]
            imgItem['annotations'] = annos
            self.imgIds.append(imgItem['image_id'])
            self.images[imgItem['image_id']] = imgItem

        assert len(self.imgIds) > 0, f'{self.ann_file} is None file'
        if self.subset == 'train':
            self.imgIds = self.imgIds[:int(len(self.imgIds) * 0.75)]
        else:
            self.imgIds = self.imgIds[int(len(self.imgIds) * 0.75):]

        self.num_images = len(self.imgIds)

    def _oc_load_keypoint_annotation_kernel(self, img_id,
                                            maxIouRange=(0., 1.)):
        """load annotation from OCHumanAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """

        data = self.images[img_id]
        file_name = data['file_name']
        width = data['width']
        height = data['height']
        num_joints = self.ann_info['num_joints']

        bbox_id = 0
        rec = []
        for i, anno in enumerate(data['annotations']):
            kpt = anno['keypoints']
            max_iou = anno['max_iou']
            if max_iou < maxIouRange[0] or max_iou >= maxIouRange[1]:
                continue
            # coco box: xyxy -> xywh
            x1, y1, x2, y2 = anno['bbox']
            x, y, w, h = [x1, y1, x2 - x1, y2 - y1]
            area = (x2 - x1) * (y2 - y1)
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if area > 0 and x2 > x1 and y2 > y1:
                bbox = [x1, y1, x2 - x1, y2 - y1]

            # coco kpt: vis 2, not vis 1, missing 0.
            # 'keypoint_visible': {'missing': 0, 'vis': 1, 'self_occluded': 2, 'others_occluded': 3},
            kptDef = self.human['keypoint_names']

            kptDefCoco = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            kptCoco = []
            num_keypoints = 0
            for i in range(len(kptDefCoco)):
                idx = kptDef.index(kptDefCoco[i])
                x, y, v = kpt[idx * 3:idx * 3 + 3]
                if v == 1 or v == 2:
                    v = 2
                    num_keypoints += 1
                elif v == 3:
                    v = 1
                    num_keypoints += 1
                kptCoco += [x, y, v]
            assert len(kptCoco) == 17 * 3

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(kptCoco).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])
            center, scale = super()._xywh2cs(*bbox)
            # image path
            image_file = os.path.join(self.img_prefix, file_name)
            rec.append({
                'image_file': image_file,
                'image_id': img_id,
                'center': center,
                'scale': scale,
                'bbox': bbox,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        return rec
