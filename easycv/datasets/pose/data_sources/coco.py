# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/top_down/topdown_coco_dataset.py
import logging
import os, wget

import json_tricks as json
import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.framework.errors import ValueError
from .top_down import PoseTopDownSource

COCO_DATASET_INFO = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and '
        'Hays, James and Perona, Pietro and Ramanan, Deva and '
        'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/'),
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
    ],
    coco2017=(
                    [
                        'http://images.cocodataset.org/zips/train2017.zip',
                        'http://images.cocodataset.org/zips/val2017.zip',
                        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                    ],
                    "unzip -d",
                    True,
                    "COCO2017"
                )
    )

DATASET_HOME = os.path.expanduser("~/.cache/easycv/dataset")


def download_file(data_cfg, dataset_home=DATASET_HOME):
    '''
    data_cfg: download config
    dataset_home: data root path
    '''

    os.makedirs(dataset_home, exist_ok=True)
    download_finished = list()
    tmp_data = data_cfg[3]
    for link_list in data_cfg[0]:
        filename = wget.filename_from_url(link_list)
        download_finished.append(filename)
        if not os.path.exists(os.path.join(dataset_home, filename)):
            try:
                print(f"{filename} is start download........")
                logging.info(f"{filename} is start download........")
                filename = wget.download(link_list, out=dataset_home)
                print(f"{filename} is download finished..........\n")
                logging.info(f"{filename} is download finished.........")

            except:
                print(f"{filename} is download fail\n")
                logging.info(f"{filename} is download fail")
                exit()

        # The prevention of Ctrol + C
        if not os.path.exists(os.path.join(dataset_home, filename)):
            exit()

    if os.path.exists(os.path.join(dataset_home, tmp_data)):
        return os.path.join(dataset_home, tmp_data)

    for tmp_file in download_finished:
        if data_cfg[2]:
            save_dir = os.path.join(dataset_home, tmp_data)
            os.makedirs(save_dir, exist_ok=True)
            if tmp_file.endswith('zip'):
                cmd = f"{data_cfg[1]} {save_dir} {os.path.join(dataset_home, tmp_file)}"
            else:
                cmd = f"{data_cfg[1]} {os.path.join(dataset_home, tmp_file)} -C {save_dir}"
        else:
            cmd = f"{data_cfg[1]} {os.path.join(dataset_home, tmp_file)} -C {dataset_home}"

        print("begin Unpack.....................")
        logging.info('begin Unpack.....................')
        os.system(cmd)
        print(f"Unpack is finished. data is saved of {dataset_home, data_cfg[3]}")
        logging.info(f"Unpack is finished. data is saved of {dataset_home, data_cfg[3]}")

    return os.path.join(dataset_home, data_cfg[3])



@DATASOURCES.register_module()
class PoseTopDownSourceCoco(PoseTopDownSource):
    """CocoSource for top-down pose estimation.

    `Microsoft COCO: Common Objects in Context' ECCV'2014
    More details can be found in the `paper
    <https://arxiv.org/abs/1405.0312>`__ .

    The source loads raw features to build a data meta object
    containing the image info, annotation info and others.

    COCO keypoint indexes::

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
        dataset_info (DatasetInfo): A class containing all dataset info.
        data_name: need download data from internet else None example coco2017
        split: train or val
        dataset_home: dataset root path, defalut: ~/.cache/easycv/dataset
        test_mode (bool): Store True when building test or
        validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 dataset_info=None,
                 data_name=None,
                 split='train',
                 dataset_home=DATASET_HOME,
                 test_mode=False):

        if dataset_info is None:
            logging.info(
                'dataset_info is missing, use default coco dataset info')
            dataset_info = COCO_DATASET_INFO

        download_cfg = COCO_DATASET_INFO.get(data_name, 0)
        # Determine whether to download it
        if download_cfg:
            root_path = download_file(download_cfg, dataset_home)
            assert split in ["train", 'val'], f"{split} not in [train, val]"
            if split == "train":
                ann_file = os.path.join(root_path, "annotations/person_keypoints_train2017.json")
                img_prefix = os.path.join(root_path, "train2017")
            else:
                ann_file = os.path.join(root_path, "annotations/person_keypoints_val2017.json")
                img_prefix = os.path.join(root_path, "val2017")

        self.use_gt_bbox = data_cfg.get('use_gt_bbox', True)
        self.bbox_file = data_cfg.get('bbox_file', None)
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            dataset_info=dataset_info,
            test_mode=test_mode)

    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_person_detection_results(self):
        """Load coco person detection results."""
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        bbox_id = 0
        for det_res in all_boxes:
            if det_res['category_id'] != 1:
                continue

            image_file = os.path.join(self.img_prefix,
                                      self.id2name[det_res['image_id']])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.det_bbox_thr:
                continue

            center, scale = self._xywh2cs(*box[:4])
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float32)
            kpt_db.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'bbox': box[:4],
                'bbox_score': score,
                'dataset': self.dataset_name,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1
        print(f'=> Total boxes after filter '
              f'low score@{self.det_bbox_thr}: {bbox_id}')
        return kpt_db
