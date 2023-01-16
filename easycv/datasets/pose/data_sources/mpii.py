# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from torchvision.datasets.utils import download_and_extract_archive

from easycv.datasets.registry import DATASOURCES
from easycv.utils.constant import CACHE_DIR
from .top_down import PoseTopDownSource

MPII_DATASET_INFO = dict(
    dataset_name='MPII',
    paper_info=dict(
        author=
        'Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt',
        title=
        '2D Human Pose Estimation: New Benchmark and State of the Art Analysis',
        container=
        'IEEE Conference on Computer Vision and Pattern Recognition (CVPR)',
        year='2014',
        homepage='http://human-pose.mpi-inf.mpg.de/'),
    keypoint_info={
        0:
        dict(
            name='right_ankle',
            id=0,
            color=[51, 153, 255],
            type='lower',
            swap='right_knee'),
        1:
        dict(
            name='right_knee',
            id=1,
            color=[51, 153, 255],
            type='lower',
            swap='right_hip'),
        2:
        dict(
            name='right_hip',
            id=2,
            color=[51, 153, 255],
            type='lower',
            swap='left_hip'),
        3:
        dict(
            name='left_hip',
            id=3,
            color=[51, 153, 255],
            type='lower',
            swap='left_knee'),
        4:
        dict(
            name='left_knee',
            id=4,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        5:
        dict(
            name='left_ankle',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='pelvis'),
        6:
        dict(
            name='pelvis',
            id=6,
            color=[255, 128, 0],
            type='lower',
            swap='thorax'),
        7:
        dict(name='thorax', id=7, color=[0, 255, 0], type='upper', swap=''),
        8:
        dict(
            name='neck', id=8, color=[255, 128, 0], type='upper', swap='head'),
        9:
        dict(
            name='head',
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
            swap=''),
        11:
        dict(
            name='right_elbow',
            id=11,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        12:
        dict(
            name='right_shoulder',
            id=12,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        13:
        dict(
            name='left_shoulder',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        14:
        dict(
            name='left_elbow',
            id=14,
            color=[255, 128, 0],
            type='upper',
            swap='right_elbow'),
        15:
        dict(
            name='left_wrist', id=15, color=[0, 255, 0], type='upper', swap='')
    },
    skeleton_info={
        0:
        dict(link=('right_ankle', 'right_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('right_knee', 'right_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_hip', 'left_hip'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('left_hip', 'left_knee'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('right_knee', 'left_ankle'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('left_ankle', 'pelvis'), id=5, color=[51, 153, 255]),
        6:
        dict(link=('pelvis', 'thorax'), id=6, color=[51, 153, 255]),
        7:
        dict(link=('right_knee', 'left_elbow'), id=7, color=[51, 153, 255]),
        8:
        dict(link=('left_elbow', 'right_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(link=('right_elbow', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(
            link=('right_elbow', 'right_shoulder'), id=11, color=[255, 128,
                                                                  0]),
        12:
        dict(
            link=('right_shoulder', 'left_shoulder'),
            id=12,
            color=[51, 153, 255]),
        13:
        dict(link=('left_elbow', 'neck'), id=13, color=[51, 153, 255]),
        14:
        dict(link=('neck', 'head'), id=14, color=[51, 153, 255]),
        15:
        dict(link=('head', 'right_wrist'), id=15, color=[51, 153, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5
    ],
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089
    ])


@DATASOURCES.register_module
class PoseTopDownSourceMpii(PoseTopDownSource):
    """Oc Human Source for top-down pose estimation.

    `Pose2Seg: Detection Free Human Instance Segmentation' ECCV'2019
    More details can be found in the `paper
    <https://arxiv.org/abs/1803.10683>`__ .

    The source loads raw features to build a data meta object
    containing the image info, annotation info and others.

    Oc Human keypoint indexes::

        0: 'right_ankle',
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'right_ear',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'neck',
        9: 'head',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'

    Args:
        data_cfg (dict): config
        path: This parameter is optional. If download is True and path is not provided,
            a temporary directory is automatically created for downloading
        download: If the value is True, the file is automatically downloaded to the path directory.
            If False, automatic download is not supported and data in the path is used
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or

    """
    _download_url_ = {
        'annotaitions':
        'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip',
        'images':
        'https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz'
    }

    def __init__(self,
                 data_cfg,
                 path=CACHE_DIR,
                 download=False,
                 dataset_info=None,
                 test_mode=False,
                 **kwargs):

        if dataset_info is None:
            logging.info(
                'dataset_info is missing, use default coco dataset info')
            dataset_info = MPII_DATASET_INFO

        self._base_folder = Path(path) / 'mpii'
        if kwargs.get('cfg', 0):
            self._download_url_ = kwargs['cfg']
        if download:
            self.download()

        ann_file = self._base_folder / 'mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
        img_prefix = self._base_folder / 'images'

        if ann_file.exists() and img_prefix.is_dir():
            super().__init__(
                ann_file,
                img_prefix,
                data_cfg,
                coco_style=False,
                dataset_info=dataset_info,
                test_mode=test_mode)

    def _get_db(self):
        """Load dataset."""
        # ground truth bbox
        gt_db = self._load_keypoint_annotations()

        return gt_db

    def _load_keypoint_annotations(self):
        self._load_mat_mpii()
        gt_db = list()
        for img_id, img_name, annorect in zip(self.img_ids, self.file_name,
                                              self.data_annorect):
            gt_db.extend(
                self._mpii_load_keypoint_annotation_kernel(
                    img_id, img_name, annorect))
        return gt_db

    def _load_mat_mpii(self):
        self.mpii = loadmat(self.ann_file)
        train_val = self.mpii['RELEASE']['img_train'][0, 0][0]

        image_id = np.argwhere(train_val == 1)

        # Name of the image corresponding to the data
        file_name = self.mpii['RELEASE']['annolist'][0,
                                                     0][0]['image'][image_id]

        data_annorect = self.mpii['RELEASE']['annolist'][
            0, 0][0]['annorect'][image_id]

        self.img_ids = self.deal_annolist(data_annorect, 'annopoints')
        self.num_images = len(self.img_ids)

        self.data_annorect = data_annorect[self.img_ids]
        self.file_name = file_name[self.img_ids]

    def _mpii_load_keypoint_annotation_kernel(self, img_id, img_file_name,
                                              annorect):
        """
        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_path = img_file_name[0]['name'][0, 0][0]
        num_joints = self.ann_info['num_joints']

        bbox_id = 0
        rec = []
        for scale, objpos, points in zip(annorect[0]['scale'][0, :],
                                         annorect[0]['objpos'][0, :],
                                         annorect[0]['annopoints'][0, :]):
            if not all(h.shape == (1, 1) for h in [scale, objpos, points]):
                continue
            if not all(k in points['point'][0, 0].dtype.fields
                       for k in ['is_visible', 'x', 'y', 'id']):
                continue

            info = self.load_points_bbox(scale, objpos, points)

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(info['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*info['bbox'])
            image_file = os.path.join(self.img_prefix, img_path)
            rec.append({
                'image_file': image_file,
                'image_id': img_id,
                'center': center,
                'scale': scale,
                'bbox': info['bbox'],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        return rec

    def load_points_bbox(self, scale, objpos, points):
        bbox = [
            objpos[0, 0]['x'][0, 0], objpos[0, 0]['y'][0, 0],
            int((scale[0, 0] * 200)),
            int((scale[0, 0] * 200))
        ]  # x,y, w, h
        bbox = [
            int(bbox[0] - bbox[2] / 2),
            int(bbox[1] - bbox[3] / 2), bbox[2], bbox[3]
        ]

        joints_3d = [0] * 3 * 16
        for x, y, d, vis in zip(points['point'][0, 0]['x'][0],
                                points['point'][0, 0]['y'][0],
                                points['point'][0, 0]['id'][0],
                                points['point'][0, 0]['is_visible'][0]):
            d = d[0, 0] * 3
            joints_3d[d] = x[0, 0]
            joints_3d[d + 1] = y[0, 0]
            if vis.shape == (1, 1):
                joints_3d[d + 2] = vis[0, 0]
            else:
                joints_3d[d + 2] = 0
        return {'bbox': bbox, 'keypoints': joints_3d}

    # Delete data without a key point
    def deal_annolist(self, num_list, char):
        num = list()
        for i, _ in enumerate(num_list):
            ids = _[0].dtype
            if len(ids) == 0:
                continue
            else:
                if char in ids.fields.keys():
                    num.append(i)
                else:
                    continue
        return num

    def download(self):

        if os.path.exists(self._base_folder):
            return self._base_folder

        # Download and extract
        for url in self._download_url_.values():
            download_and_extract_archive(
                url,
                str(self._base_folder),
                str(self._base_folder),
                remove_finished=True)
