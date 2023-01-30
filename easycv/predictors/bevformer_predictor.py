# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from easycv.core.bbox import get_box_type
from easycv.datasets.registry import PIPELINES
from easycv.datasets.shared.pipelines.format import to_tensor
from easycv.datasets.shared.pipelines.transforms import Compose
from easycv.framework.errors import ValueError
from easycv.predictors.base import InputProcessor, PredictorV2
from easycv.predictors.builder import PREDICTORS
from easycv.utils.misc import encode_str_to_tensor
from easycv.utils.registry import build_from_cfg


class BEVFormerInputProcessor(InputProcessor):
    """Process inputs for BEVFormer model.

    Args:
        cfg (Config): Config instance.
        pipelines (list[dict]): Data pipeline configs.
        batch_size (int): batch size for forward.
        use_camera (bool): Whether use camera data.
        box_type_3d (str): Box type.
        threads (int): Number of processes to process inputs.
    """

    def __init__(self,
                 cfg,
                 pipelines=None,
                 batch_size=1,
                 use_camera=True,
                 box_type_3d='LiDAR',
                 adapt_jit=False,
                 threads=8):
        self.use_camera = use_camera
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.adapt_jit = adapt_jit

        super(BEVFormerInputProcessor, self).__init__(
            cfg, pipelines=pipelines, batch_size=batch_size, threads=threads)

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

    def process_single(self, input):
        """Process single input sample.
        Args:
            input (str): Pickle file path, the content format is the same with the infos file of nusences.
        """
        data_info = mmcv.load(input) if isinstance(input, str) else input
        result = self._prepare_input_dict(data_info)
        result = self.processor(result)

        if self.adapt_jit:
            result['can_bus'] = DC(
                to_tensor(result['img_metas'][0]._data['can_bus']),
                cpu_only=False)
            result['lidar2img'] = DC(
                to_tensor(result['img_metas'][0]._data['lidar2img']),
                cpu_only=False)
            result['scene_token'] = DC(
                torch.tensor(
                    bytearray(
                        pickle.dumps(
                            result['img_metas'][0]._data['scene_token'])),
                    dtype=torch.uint8),
                cpu_only=False)
            result['img_shape'] = DC(
                to_tensor(result['img_metas'][0]._data['img_shape']),
                cpu_only=False)
        else:
            result['can_bus'] = DC(
                torch.stack(
                    [to_tensor(result['img_metas'][0]._data['can_bus'])]),
                cpu_only=False)
            result['lidar2img'] = DC(
                torch.stack(
                    [to_tensor(result['img_metas'][0]._data['lidar2img'])]),
                cpu_only=False)

        return result


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
        input_processor_threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
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
                 model_type=None,
                 input_processor_threads=8,
                 mode='BGR',
                 *arg,
                 **kwargs):
        if batch_size > 1:
            raise ValueError(
                f'Only support batch_size=1 now, but get batch_size={batch_size}'
            )
        self.model_type = model_type
        if self.model_type is None:
            if model_path.endswith('jit'):
                self.model_type = 'jit'
            elif model_path.endswith('blade'):
                self.model_type = 'blade'
        self.is_jit_model = self.model_type in ['jit', 'blade']
        self.use_camera = use_camera
        self.score_threshold = score_threshold
        self.result_key = 'pts_bbox'
        self.box_type_3d_str = box_type_3d
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        super(BEVFormerPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=input_processor_threads,
            mode=mode,
            *arg,
            **kwargs)

        self.CLASSES = self.cfg.get('CLASSES', None)
        # The initial prev_bev should be the weight of self.model.pts_bbox_head.bev_embedding, but the weight cannot be taken out from the blade model.
        # So we using the dummy data as the the initial value, and it will not be used, just to adapt to jit and blade models.
        # init_prev_bev = self.model.pts_bbox_head.bev_embedding.weight.clone().detach()
        # init_prev_bev = init_prev_bev[:, None, :],  # [40000, 256] -> [40000, 1, 256]
        dummy_prev_bev = torch.rand(
            [self.cfg.bev_h * self.cfg.bev_w, 1,
             self.cfg.embed_dim]).to(self.device)
        self.prev_frame_info = {
            'prev_bev': dummy_prev_bev.to(self.device),
            'prev_scene_token': encode_str_to_tensor('dummy_prev_scene_token'),
            'prev_pos': torch.tensor(0),
            'prev_angle': torch.tensor(0),
        }

    def get_input_processor(self):
        return BEVFormerInputProcessor(
            self.cfg,
            pipelines=self.pipelines,
            batch_size=self.batch_size,
            use_camera=self.use_camera,
            box_type_3d=self.box_type_3d_str,
            adapt_jit=self.is_jit_model,
            threads=self.input_processor_threads)

    def prepare_model(self):
        if self.is_jit_model:
            model = torch.jit.load(self.model_path, map_location=self.device)
            return model
        return super().prepare_model()

    def model_forward(self, inputs):
        if self.is_jit_model:
            with torch.no_grad():
                img = inputs['img'][0][0]
                img_metas = {
                    'can_bus': inputs['can_bus'][0],
                    'lidar2img': inputs['lidar2img'][0],
                    'img_shape': inputs['img_shape'][0],
                    'scene_token': inputs['scene_token'][0],
                    'prev_bev': self.prev_frame_info['prev_bev'],
                    'prev_pos': self.prev_frame_info['prev_pos'],
                    'prev_angle': self.prev_frame_info['prev_angle'],
                    'prev_scene_token':
                    self.prev_frame_info['prev_scene_token']
                }
                inputs = (img, img_metas)
                outputs = self.model(*inputs)

            # update prev_frame_info
            self.prev_frame_info['prev_bev'] = outputs[3][0]
            self.prev_frame_info['prev_pos'] = outputs[3][1]
            self.prev_frame_info['prev_angle'] = outputs[3][2]
            self.prev_frame_info['prev_scene_token'] = outputs[3][3]

            outputs = {
                'pts_bbox': [{
                    'scores_3d':
                    outputs[0],
                    'labels_3d':
                    outputs[1],
                    'boxes_3d':
                    self.box_type_3d(outputs[2].cpu(), outputs[2].size()[-1])
                }],
            }
            return outputs
        return super().model_forward(inputs)

    def visualize(self, inputs, results, out_dir, show=False, pipeline=None):
        raise NotImplementedError
