# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import concurrent.futures
import copy
import logging
import random
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from easycv.core.bbox import Box3DMode, Coord3DMode
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset
from easycv.datasets.shared.pipelines import Compose
from easycv.datasets.shared.pipelines.format import to_tensor
from .utils import extract_result_dict


@DATASETS.register_module
class NuScenesDataset(BaseDataset):
    """Dataset for NuScenes.
    """

    def __init__(self,
                 data_source,
                 pipeline,
                 queue_length=1,
                 eval_version='detection_cvpr_2019',
                 profiling=False):
        """
        Args:
            data_source: Data_source config dict
            pipeline: Pipeline config list
            queue_length: Each sequence contains `queue_length` frames.
            eval_version (bool, optional): Configuration version of evaluation.
                Defaults to  'detection_cvpr_2019'.
            profiling: If set True, will print pipeline time
        """
        super(NuScenesDataset, self).__init__(
            data_source, pipeline, profiling=profiling)

        self.queue_length = queue_length
        self.CLASSES = self.data_source.CLASSES
        self.with_velocity = self.data_source.with_velocity
        self.modality = self.data_source.modality

        from nuscenes.eval.detection.config import config_factory
        self.eval_version = eval_version
        self.eval_detection_configs = config_factory(self.eval_version)
        self.flag = np.zeros(
            len(self), dtype=np.uint8)  # for DistributedGroupSampler
        self.pipeline_cfg = pipeline

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_source.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(
                self.data_source.data_infos[sample_id], boxes,
                mapped_class_names, self.eval_detection_configs)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.data_source.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.data_source.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 evaluators=[],
                 logger=None,
                 jsonfile_prefix=None,
                 **kwargs):
        """Evaluation in nuScenes protocol.

        Args:
            results (dict[list]): Testing results of the dataset.
            evaluators: Evaluators to calculate metric with results and groundtruth.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        from nuscenes import NuScenes

        results_list = [{} for _ in range(len(self))]
        for k, v in results.items():
            assert isinstance(v, list)
            for i, result in enumerate(v):
                results_list[i].update({k: result})

        del results
        result_files, tmp_dir = self.format_results(results_list,
                                                    jsonfile_prefix)
        nusc = NuScenes(
            version=self.data_source.version,
            dataroot=self.data_source.data_root,
            verbose=True)

        results_dict = {}
        for evaluator in evaluators:
            results_dict.update(
                evaluator.evaluate(
                    result_files,
                    nusc,
                    eval_detection_configs=self.eval_detection_configs))

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            return self._build_default_pipeline()
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        input_dict = self.data_source[index]
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]

        return data

    def visualize(self, results, out_dir, show=False, pipeline=None, **kwargs):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_source.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.data_source.get_ann_info(
                i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            from easycv.core.visualization.image_3d import show_result
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None

        can_bus_list = []
        lidar2img_list = []
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

            can_bus_list.append(to_tensor(metas_map[i]['can_bus']))
            lidar2img_list.append(to_tensor(metas_map[i]['lidar2img']))

        queue[-1]['img'] = DC(
            torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue[-1]['can_bus'] = DC(torch.stack(can_bus_list), cpu_only=False)
        queue[-1]['lidar2img'] = DC(
            torch.stack(lidar2img_list), cpu_only=False)
        queue = queue[-1]
        return queue

    @staticmethod
    def _get_single_data(i,
                         data_source,
                         pipeline,
                         flip_flag=False,
                         scale=None):
        i = max(0, i)
        try:
            data = data_source[i]
            data['flip_flag'] = flip_flag
            if scale:
                data['resize_scale'] = scale
            data = pipeline(data)
            if data is None or ~(data['gt_labels_3d']._data != -1).any():
                return None
        except Exception as e:
            logging.error(e)
            return None
        return i, data

    def _get_queue_data(self, idx):
        queue = []
        idx_list = list(range(idx - self.queue_length, idx))
        random.shuffle(idx_list)
        idx_list = sorted(idx_list[1:])
        idx_list.append(idx)

        flip_flag = False
        scale = None
        for member in self.pipeline_cfg:

            if member['type'] == 'RandomScaleImageMultiViewImage':
                scales = member['scales']
                rand_ind = np.random.permutation(range(len(scales)))[0]
                scale = scales[rand_ind]
            if member['type'] == 'RandomHorizontalFlipMultiViewImage':
                flip_flag = np.random.rand() >= 0.5

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(idx_list)) as executor:
            threads = []
            for i in idx_list:
                future = executor.submit(self._get_single_data, i,
                                         self.data_source, self.pipeline,
                                         flip_flag, scale)
                threads.append(future)

            for future in concurrent.futures.as_completed(threads):
                queue.append(future.result())

        if None in queue:
            return None

        queue = sorted(queue, key=lambda item: item[0])
        queue = [item[1] for item in queue]

        return self.union2one(queue)

    def __getitem__(self, idx):
        while True:
            if self.queue_length > 1:
                data_dict = self._get_queue_data(idx)
            else:
                data_dict = self.data_source[idx]
                data_dict = self.pipeline(data_dict)

                can_bus_list, lidar2img_list = [], []
                for i in range(len(data_dict['img_metas'])):
                    can_bus_list.append(
                        to_tensor(data_dict['img_metas'][i]._data['can_bus']))
                    lidar2img_list.append(
                        to_tensor(
                            data_dict['img_metas'][i]._data['lidar2img']))
                data_dict['can_bus'] = DC(
                    torch.stack(can_bus_list), cpu_only=False)
                data_dict['lidar2img'] = DC(
                    torch.stack(lidar2img_list), cpu_only=False)

            if data_dict is None:
                idx = self._rand_another(idx)
                continue
            return data_dict


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    import pyquaternion
    from nuscenes.utils.data_classes import Box as NuScenesBox

    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info, boxes, classes, eval_configs):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    import pyquaternion

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
