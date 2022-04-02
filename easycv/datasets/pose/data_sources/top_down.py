# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/base/kpt_2d_sview_rgb_img_top_down_dataset.py
import copy
import logging
import os
from abc import ABCMeta

import mmcv
import numpy as np
from mmcv import Config
from mmcv.utils.path import is_filepath
from xtcocotools.coco import COCO

from easycv.datasets.registry import DATASOURCES


class DatasetInfo:

    def __init__(self, dataset_info):
        self._dataset_info = dataset_info
        self.dataset_name = self._dataset_info.get('dataset_name', '')
        self.keypoint_info = self._dataset_info['keypoint_info']
        self.skeleton_info = self._dataset_info['skeleton_info']
        self.joint_weights = np.array(
            self._dataset_info['joint_weights'], dtype=np.float32)[:, None]

        self.sigmas = np.array(self._dataset_info['sigmas'])

        self._parse_keypoint_info()
        self._parse_skeleton_info()

    def _parse_skeleton_info(self):
        """Parse skeleton information.

        - link_num (int): number of links.
        - skeleton (list((2,))): list of links (id).
        - skeleton_name (list((2,))): list of links (name).
        - pose_link_color (np.ndarray): the color of the link for
            visualization.
        """
        self.link_num = len(self.skeleton_info.keys())
        self.pose_link_color = []

        self.skeleton_name = []
        self.skeleton = []
        for skid in self.skeleton_info.keys():
            link = self.skeleton_info[skid]['link']
            self.skeleton_name.append(link)
            self.skeleton.append([
                self.keypoint_name2id[link[0]], self.keypoint_name2id[link[1]]
            ])
            self.pose_link_color.append(self.skeleton_info[skid].get(
                'color', [255, 128, 0]))
        self.pose_link_color = np.array(self.pose_link_color)

    def _parse_keypoint_info(self):
        """Parse keypoint information.

        - keypoint_num (int): number of keypoints.
        - keypoint_id2name (dict): mapping keypoint id to keypoint name.
        - keypoint_name2id (dict): mapping keypoint name to keypoint id.
        - upper_body_ids (list): a list of keypoints that belong to the
            upper body.
        - lower_body_ids (list): a list of keypoints that belong to the
            lower body.
        - flip_index (list): list of flip index (id)
        - flip_pairs (list((2,))): list of flip pairs (id)
        - flip_index_name (list): list of flip index (name)
        - flip_pairs_name (list((2,))): list of flip pairs (name)
        - pose_kpt_color (np.ndarray): the color of the keypoint for
            visualization.
        """

        self.keypoint_num = len(self.keypoint_info.keys())
        self.keypoint_id2name = {}
        self.keypoint_name2id = {}

        self.pose_kpt_color = []
        self.upper_body_ids = []
        self.lower_body_ids = []

        self.flip_index_name = []
        self.flip_pairs_name = []

        for kid in self.keypoint_info.keys():

            keypoint_name = self.keypoint_info[kid]['name']
            self.keypoint_id2name[kid] = keypoint_name
            self.keypoint_name2id[keypoint_name] = kid
            self.pose_kpt_color.append(self.keypoint_info[kid].get(
                'color', [255, 128, 0]))

            type = self.keypoint_info[kid].get('type', '')
            if type == 'upper':
                self.upper_body_ids.append(kid)
            elif type == 'lower':
                self.lower_body_ids.append(kid)
            else:
                pass

            swap_keypoint = self.keypoint_info[kid].get('swap', '')
            if swap_keypoint == keypoint_name or swap_keypoint == '':
                self.flip_index_name.append(keypoint_name)
            else:
                self.flip_index_name.append(swap_keypoint)
                if [swap_keypoint, keypoint_name] not in self.flip_pairs_name:
                    self.flip_pairs_name.append([keypoint_name, swap_keypoint])

        self.flip_pairs = [[
            self.keypoint_name2id[pair[0]], self.keypoint_name2id[pair[1]]
        ] for pair in self.flip_pairs_name]
        self.flip_index = [
            self.keypoint_name2id[name] for name in self.flip_index_name
        ]
        self.pose_kpt_color = np.array(self.pose_kpt_color)


@DATASOURCES.register_module()
class PoseTopDownSource(object, metaclass=ABCMeta):
    """Class for keypoint 2D top-down pose estimation with
    single-view RGB image as the data source.

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        dataset_info (DatasetInfo): A class containing all dataset info.
        coco_style (bool): Whether the annotation json is coco-style.
            Default: True
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 dataset_info,
                 coco_style=True,
                 test_mode=False):

        if not coco_style:
            raise ValueError('Only support `coco_style` now!')
        if is_filepath(dataset_info):
            cfg = Config.fromfile(dataset_info)
            dataset_info = cfg._cfg_dict['dataset_info']

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg.get(
            'dataset_channel', None)
        self.ann_info['inference_channel'] = data_cfg.get(
            'inference_channel', None)

        self.ann_info['use_different_joint_weights'] = data_cfg.get(
            'use_different_joint_weights', False)

        dataset_info = DatasetInfo(dataset_info)

        assert self.ann_info['num_joints'] == dataset_info.keypoint_num
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.ann_info['flip_index'] = dataset_info.flip_index
        self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        self.ann_info['skeleton'] = dataset_info.skeleton
        self.sigmas = dataset_info.sigmas
        self.dataset_name = dataset_info.dataset_name

        if coco_style:
            self.coco = COCO(ann_file)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

        self.db = self._get_db()
        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if (not self.test_mode) and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * padding

        return center, scale

    def _get_db(self):
        """Load dataset."""
        # ground truth bbox
        gt_db = self._load_keypoint_annotations()

        return gt_db

    def _load_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_keypoint_annotation_kernel(img_id))
        return gt_db

    def _load_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'image_id': img_id,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': bbox_id
            })
            bbox_id = bbox_id + 1

        return rec

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def load_image(self, image_file):
        img = mmcv.imread(image_file, 'color', 'rgb')
        return img

    def get_length(self):
        """Get the size of the dataset."""
        return len(self.db)

    def get_sample(self, idx):
        results = copy.deepcopy(self.db[idx])
        # TODO: optimize fault tolerance for image load
        try:
            img = self.load_image(results['image_file'])
            results['img'] = img
        except:
            logging.warning('Fail to read %s%s, load next sample!' %
                            (idx, results['image_file']))
            return self.get_sample((idx + 1) % self.get_length())

        results['ann_info'] = self.ann_info

        return results
