# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import mmcv
import numpy as np
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

from easycv.core.bbox import Box3DMode, Coord3DMode, get_box_type
from easycv.datasets.detection3d.utils import extract_result_dict
from easycv.datasets.registry import PIPELINES
from easycv.datasets.shared.pipelines.transforms import Compose
from easycv.utils.registry import build_from_cfg
from .base import PredictorV2
from .builder import PREDICTORS


@PREDICTORS.register_module()
class BEVFormerPredictor(PredictorV2):
    """Predictor for BEVFormer.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 box_type_3d='LiDAR',
                 use_camera=True,
                 score_threshold=0.1,
                 *arg,
                 **kwargs):
        super(BEVFormerPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            *arg,
            **kwargs)
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.CLASSES = self.cfg.get('CLASSES', None)
        self.use_camera = use_camera
        self.score_threshold = score_threshold
        self.result_key = 'pts_bbox'

    def _prepare_input_dict(self, data_info):
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=data_info['token'],
            pts_filename=data_info['lidar_path'],
            sweeps=data_info['sweeps'],
            ego2global_translation=data_info['ego2global_translation'],
            ego2global_rotation=data_info['ego2global_rotation'],
            prev_idx=data_info['prev'],
            next_idx=data_info['next'],
            scene_token=data_info['scene_token'],
            can_bus=data_info['can_bus'],
            frame_idx=data_info['frame_idx'],
            timestamp=data_info['timestamp'] / 1e6,
        )

        if self.use_camera:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in data_info['cams'].items():
                cam_info['data_path'] = os.path.expanduser(
                    cam_info['data_path'])
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        input_dict['img_fields'] = []
        input_dict['bbox3d_fields'] = []
        input_dict['pts_mask_fields'] = []
        input_dict['pts_seg_fields'] = []
        input_dict['bbox_fields'] = []
        input_dict['mask_fields'] = []
        input_dict['seg_fields'] = []
        input_dict['box_type_3d'] = self.box_type_3d
        input_dict['box_mode_3d'] = self.box_mode_3d

        load_pipelines = [
            dict(type='LoadMultiViewImageFromFiles', to_float32=True)
        ]
        load_pipelines = Compose(
            [build_from_cfg(p, PIPELINES) for p in load_pipelines])
        result = load_pipelines(input_dict)
        return result

    def preprocess_single(self, input):
        """Preprocess single input sample.
        If you need custom ops to load or process a single input sample, you need to reimplement it.
        """
        data_info = mmcv.load(input)
        result = self._prepare_input_dict(data_info)
        return self.processor(result)

    def postprocess_single(self, inputs, *args, **kwargs):
        # TODO: filter results by score_threshold
        return super().postprocess_single(inputs, *args, **kwargs)

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

    def _extract_data(self, data_info, pipeline, key):
        """Load data using input pipeline and extract data according to key.

        Args:
            data_info (int): Data info load from input file.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        input_dict = self._prepare_input_dict(data_info)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]

        return data

    def visualize(self, inputs, results, out_dir, show=False, pipeline=None):
        from easycv.core.visualization.image_3d import show_result

        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)

        for i, input_i in enumerate(inputs):
            data_info = mmcv.load(input_i)
            result = results[i]
            if self.result_key in result.keys():
                result = result[self.result_key]
            pts_path = data_info['lidar_path']
            file_name = os.path.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(data_info, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > self.score_threshold
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(
                points,
                gt_bboxes=None,
                pred_bboxes=show_pred_bboxes,
                out_dir=out_dir,
                filename=file_name,
                show=show)
