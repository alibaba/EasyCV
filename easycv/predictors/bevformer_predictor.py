# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import mmcv
import numpy as np

from easycv.core.bbox import get_box_type
from easycv.datasets.registry import PIPELINES
from easycv.datasets.shared.pipelines.transforms import Compose
from easycv.utils.registry import build_from_cfg
from .base import PredictorV2
from .builder import PREDICTORS


@PREDICTORS.register_module()
class BEVFormerPredictor(PredictorV2):
    """Predictor for BEVFormer.

    Args:
        model_path (str): Path of model path.
        config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
        batch_size (int): batch size for forward.
        device (str | torch.device): Support str('cuda' or 'cpu') or torch.device, if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        box_type_3d (str): Box type.
        use_camera (bool): Whether use camera data.
        score_threshold (float): Score threshold to filter inference results.
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
        from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

        input_dict = dict(
            ego2global_translation=data_info['ego2global_translation'],
            ego2global_rotation=data_info['ego2global_rotation'],
            scene_token=data_info['scene_token'],
            can_bus=data_info['can_bus'])
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
        Args:
            input (str): Pickle file path, the content format is the same with the infos file of nusences.
        """
        data_info = mmcv.load(input)
        result = self._prepare_input_dict(data_info)
        return self.processor(result)

    def postprocess_single(self, inputs, *args, **kwargs):
        # TODO: filter results by score_threshold
        return super().postprocess_single(inputs, *args, **kwargs)

    def visualize(self, inputs, results, out_dir, show=False, pipeline=None):
        raise NotImplementedError
