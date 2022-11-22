# Modified from https://github.com/fundamentalvision/BEVFormer.
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import pickle

import numpy as np
import torch

from easycv.core.bbox import get_box_type
from easycv.core.bbox.bbox_util import bbox3d2result
from easycv.models.detection3d.detectors.mvx_two_stage import \
    MVXTwoStageDetector
from easycv.models.detection3d.utils.grid_mask import GridMask
from easycv.models.registry import MODELS
from easycv.utils.misc import decode_tensor_to_str, encode_str_to_tensor


@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
        extract_feat_serially (bool): Whether extract history features one by one,
            to solve the problem of batchnorm corrupt when shape N is too large.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 extract_feat_serially=False):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.extract_feat_serially = extract_feat_serially

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'prev_scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(
                        int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def obtain_history_bev_serially(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively.
        Extract feature one by one to solve the problem of batchnorm corrupt when shape N is too large.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            for i in range(len_queue):
                img_feats = self.extract_feat(
                    img=imgs_queue[:, i, ...], len_queue=None)
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()

            return prev_bev

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(
                img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        self._check_inputs(img_metas, img, kwargs)

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)

        if self.extract_feat_serially:
            prev_bev = self.obtain_history_bev_serially(
                prev_img, prev_img_metas)
        else:
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def _check_inputs(self, img_metas, img, kwargs):
        can_bus_in_kwargs = kwargs.get('can_bus', None) is not None
        lidar2img_in_kwargs = kwargs.get('lidar2img', None) is not None
        for batch_i in range(len(img_metas)):
            for i in range(len(img_metas[batch_i])):
                if can_bus_in_kwargs:
                    img_metas[batch_i][i]['can_bus'] = kwargs['can_bus'][
                        batch_i][i]
                else:
                    if isinstance(img_metas[batch_i][i]['can_bus'],
                                  np.ndarray):
                        img_metas[batch_i][i]['can_bus'] = torch.from_numpy(
                            img_metas[batch_i][i]['can_bus']).to(img.device)
                if lidar2img_in_kwargs:
                    img_metas[batch_i][i]['lidar2img'] = kwargs['lidar2img'][
                        batch_i][i]
                else:
                    if isinstance(img_metas[batch_i][i]['lidar2img'],
                                  np.ndarray):
                        img_metas[batch_i][i]['lidar2img'] = torch.from_numpy(
                            np.array(img_metas[batch_i][i]['lidar2img'])).to(
                                img.device)
        kwargs.pop('can_bus', None)
        kwargs.pop('lidar2img', None)

    def forward_test(self, img_metas, img=None, rescale=True, **kwargs):
        self._check_inputs(img_metas, img, kwargs)

        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info[
                'prev_scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['prev_scene_token'] = img_metas[0][0][
            'scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = img_metas[0][0]['can_bus'][:3].clone()
        tmp_angle = img_metas[0][0]['can_bus'][-1].clone()
        # skip init dummy prev_bev
        if self.prev_frame_info['prev_bev'] is not None and not torch.equal(
                self.prev_frame_info['prev_bev'],
                self.prev_frame_info['prev_bev'].new_zeros(
                    self.prev_frame_info['prev_bev'].size())):
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info[
                'prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0],
            img[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            rescale=rescale,
            **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        results_dict = {}
        for bbox_result in bbox_results:
            for result_name, results in bbox_result.items():
                if result_name not in results_dict:
                    results_dict[result_name] = []
                results_dict[result_name].append(results)

        return results_dict

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list

    def forward_export(self, img, img_metas):
        error_str = 'Only support batch_size=1 and queue_length=1, please remove axis of batch_size and queue_length!'
        if len(img.shape) > 4:
            raise ValueError(error_str)
        elif len(img.shape) < 4:
            raise ValueError(
                'The length of img size must be equal to 4: [num_cameras, img_channel, img_height, img_width]'
            )

        assert len(
            img_metas['can_bus'].shape) == 1, error_str  # torch.Size([18])
        assert len(img_metas['lidar2img'].shape
                   ) == 3, error_str  # torch.Size([6, 4, 4])
        assert len(
            img_metas['img_shape'].shape) == 2, error_str  # torch.Size([6, 3])
        assert len(img_metas['prev_bev'].shape
                   ) == 3, error_str  # torch.Size([40000, 1, 256])

        img = img[
            None, None,
            ...]  # torch.Size([6, 3, 928, 1600]) -> torch.Size([1, 1, 6, 3, 928, 1600])

        box_type_3d = img_metas.get('box_type_3d', 'LiDAR')
        if isinstance(box_type_3d, torch.Tensor):
            box_type_3d = pickle.loads(box_type_3d.cpu().numpy().tobytes())
        img_metas['box_type_3d'] = get_box_type(box_type_3d)[0]
        img_metas['scene_token'] = decode_tensor_to_str(
            img_metas['scene_token'])

        # previous frame info
        self.prev_frame_info['prev_scene_token'] = decode_tensor_to_str(
            img_metas.pop('prev_scene_token', None))
        self.prev_frame_info['prev_bev'] = img_metas.pop('prev_bev', None)
        self.prev_frame_info['prev_pos'] = img_metas.pop('prev_pos', None)
        self.prev_frame_info['prev_angle'] = img_metas.pop('prev_angle', None)

        img_metas = [[img_metas]]
        outputs = self.forward_test(img_metas, img=img)
        scores_3d = outputs['pts_bbox'][0]['scores_3d']
        labels_3d = outputs['pts_bbox'][0]['labels_3d']
        boxes_3d = outputs['pts_bbox'][0]['boxes_3d'].tensor.cpu()

        # info has been updated to the current frame
        prev_bev = self.prev_frame_info['prev_bev']
        prev_pos = self.prev_frame_info['prev_pos']
        prev_angle = self.prev_frame_info['prev_angle']
        prev_scene_token = encode_str_to_tensor(
            self.prev_frame_info['prev_scene_token'])

        return scores_3d, labels_3d, boxes_3d, [
            prev_bev, prev_pos, prev_angle, prev_scene_token
        ]

    def forward_history_bev(self,
                            img,
                            can_bus,
                            lidar2img,
                            img_shape,
                            scene_token,
                            box_type_3d='LiDAR'):
        """Experimental api, for export jit model to obtain history bev.
        """
        if isinstance(box_type_3d, torch.Tensor):
            box_type_3d = pickle.loads(box_type_3d.cpu().numpy().tobytes())

        batch_size, len_queue = img.size()[:2]
        img_metas = []
        for b_i in range(batch_size):
            img_metas.append([])
            for i in range(len_queue):
                scene_token_str = pickle.loads(
                    scene_token[b_i][i].cpu().numpy().tobytes())
                img_metas[b_i].append({
                    'scene_token':
                    scene_token_str,
                    'can_bus':
                    can_bus[b_i][i],
                    'lidar2img':
                    lidar2img[b_i][i],
                    'img_shape':
                    img_shape[b_i][i],
                    'box_type_3d':
                    get_box_type(box_type_3d)[0],
                    'prev_bev_exists':
                    False
                })

        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)

        if self.extract_feat_serially:
            prev_bev = self.obtain_history_bev_serially(
                prev_img, prev_img_metas)
        else:
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        return prev_bev
