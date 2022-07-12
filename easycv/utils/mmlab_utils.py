# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import logging

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from easycv.models.registry import BACKBONES, HEADS, MODELS, NECKS
from .test_util import run_in_subprocess

EASYCV_REGISTRY_MAP = {
    'model': MODELS,
    'backbone': BACKBONES,
    'neck': NECKS,
    'head': HEADS
}
MMDET = 'mmdet'
SUPPORT_MMLAB_TYPES = [MMDET]
_MMLAB_COPIES = locals()


class MMAdapter:

    def __init__(self, modules_config):
        """Adapt mmlab apis.
        Args: modules_config is as follow format:
            [
                dict(type='mmdet', name='MaskRCNN', module='model'), # means using mmdet MaskRCNN
                # dict(type='mmdet, name='ResNet', module='backbone'), # comment out, means use my ResNet
                dict(name='FPN', module='neck'),  # type is missing, use mmdet default
            ]
        """
        self.default_mmtype = 'mmdet'
        self.mmtype_list = set([])

        for module_cfg in modules_config:
            mmtype = module_cfg.get('type',
                                    self.default_mmtype)  # default mmdet
            self.mmtype_list.add(mmtype)

        self.check_env()
        self.fix_conflicts()

        self.MMTYPE_REGISTRY_MAP = self._get_mmtype_registry_map()
        self.modules_config = modules_config

    def check_env(self):
        assert self.mmtype_list.issubset(
            SUPPORT_MMLAB_TYPES), 'Only support %s now !' % SUPPORT_MMLAB_TYPES
        install_success = False
        try:
            import mmdet
            install_success = True
        except ModuleNotFoundError as e:
            logging.warning(e)
            logging.warning('Try to install mmdet...')

        if not install_success:
            try:
                run_in_subprocess('pip install mmdet')
            except:
                raise ValueError(
                    'Failed to install mmdet, '
                    'please refer to https://github.com/open-mmlab/mmdetection to install.'
                )

    def fix_conflicts(self):
        # mmdet and easycv both register
        if MMDET in self.mmtype_list:
            mmcv_conflict_list = ['YOLOXLrUpdaterHook']
            from mmcv.runner.hooks import HOOKS
            for conflict in mmcv_conflict_list:
                HOOKS._module_dict.pop(conflict, None)

    def adapt_mmlab_modules(self):
        for module_cfg in self.modules_config:
            mmtype = module_cfg['type']
            module_name, module_type = module_cfg['name'], module_cfg['module']
            self._merge_mmlab_module_to_easycv(mmtype, module_type,
                                               module_name)
            self.wrap_module(mmtype, module_type, module_name)

        for mmtype in self.mmtype_list:
            self._merge_all_easycv_modules_to_mmlab(mmtype)

    def wrap_module(self, mmtype, module_type, module_name):
        module_obj = self._get_mm_module_obj(mmtype, module_type, module_name)
        if mmtype == MMDET:
            MMDetWrapper().wrap_module(module_obj, module_type)

    def _merge_all_easycv_modules_to_mmlab(self, mmtype):
        # Add all my module to mmlab module registry, if duplicated, replace with my module.
        # To handle: if MaskRCNN use mmdet's api, but the backbone also uses the backbone registered in mmdet
        # In order to support our backbone, register our modules into mmdet.
        # If not specified mmdet type, use our modules by default.
        for key, registry_type in self.MMTYPE_REGISTRY_MAP[mmtype].items():
            registry_type._module_dict.update(
                EASYCV_REGISTRY_MAP[key]._module_dict)

    def _merge_mmlab_module_to_easycv(self,
                                      mmtype,
                                      module_type,
                                      module_name,
                                      force=True):
        model_obj = self._get_mm_module_obj(mmtype, module_type, module_name)
        # Add mmlab module to my module registry.
        easycv_registry_type = EASYCV_REGISTRY_MAP[module_type]
        # Copy a duplicate to avoid directly modifying the properties of the original object
        _MMLAB_COPIES[module_name] = type(module_name, (model_obj, ), dict())
        easycv_registry_type.register_module(
            _MMLAB_COPIES[module_name], force=force)

    def _get_mm_module_obj(self, mmtype, module_type, module_name):
        if isinstance(module_name, str):
            mm_registry_type = self.MMTYPE_REGISTRY_MAP[mmtype][module_type]
            mm_module_dict = mm_registry_type._module_dict
            if module_name in mm_module_dict:
                module_obj = mm_module_dict[module_name]
            else:
                raise ValueError('Not find {} object in {}'.format(
                    module_name, mmtype))
        elif inspect.isclass(module_name):
            module_obj = module_name
        else:
            raise ValueError(
                'Only support type `str` and `class` object, but get type {}'.
                format(type(module_name)))
        return module_obj

    def _get_mmtype_registry_map(self):
        from mmdet.models.builder import MODELS as MMMODELS
        from mmdet.models.builder import BACKBONES as MMBACKBONES
        from mmdet.models.builder import NECKS as MMNECKS
        from mmdet.models.builder import HEADS as MMHEADS
        registry_map = {
            MMDET: {
                'model': MMMODELS,
                'backbone': MMBACKBONES,
                'neck': MMNECKS,
                'head': MMHEADS
            }
        }
        return registry_map


class MMDetWrapper:

    def __init__(self):
        self.refactor_modules()

    def wrap_module(self, cls, module_type):
        if hasattr(cls, 'is_wrap') and cls.is_wrap:
            return
        if module_type == 'model':
            self._wrap_model_init(cls)
            self._wrap_model_forward(cls)
            self._wrap_model_forward_test(cls)
            cls.is_wrap = True

    def refactor_modules(self):
        update_rpn_head()

    def _wrap_model_init(self, cls):
        origin_init = cls.__init__

        def _new_init(self, *args, **kwargs):
            origin_init(self, *args, **kwargs)
            self.init_weights()

        setattr(cls, '__init__', _new_init)

    def _wrap_model_forward(self, cls):
        origin_forward = cls.forward

        def _new_forward(self, img, mode='train', **kwargs):
            img_metas = kwargs.pop('img_metas', None)

            if mode == 'train':
                return origin_forward(
                    self, img, img_metas, return_loss=True, **kwargs)
            else:
                return origin_forward(
                    self, img, img_metas, return_loss=False, **kwargs)

        setattr(cls, 'forward', _new_forward)

    def _wrap_model_forward_test(self, cls):
        from mmdet.core import encode_mask_results

        origin_forward_test = cls.forward_test

        def _new_forward_test(self, img, img_metas=None, **kwargs):
            kwargs.update({'rescale': True})  # move from single_gpu_test
            logging.info('Set rescale to True for `model.forward_test`!')

            result = origin_forward_test(self, img, img_metas, **kwargs)
            # ============result process to adapt to easycv============
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
            # This logic is only used in panoptic segmentation test.
            elif isinstance(result[0], dict) and 'ins_results' in result[0]:
                for j in range(len(result)):
                    bbox_results, mask_results = result[j]['ins_results']
                    result[j]['ins_results'] = (
                        bbox_results, encode_mask_results(mask_results))

            detection_boxes = []
            detection_scores = []
            detection_classes = []
            detection_masks = []
            for res_i in result:
                if isinstance(res_i, tuple):
                    bbox_result, segm_result = res_i
                    if isinstance(segm_result, tuple):
                        segm_result = segm_result[0]  # ms rcnn
                else:
                    bbox_result, segm_result = res_i, None
                bboxes = np.vstack(bbox_result)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                # draw segmentation masks
                segms = []
                if segm_result is not None and len(labels) > 0:  # non empty
                    segms = mmcv.concat_list(segm_result)
                    if isinstance(segms[0], torch.Tensor):
                        segms = torch.stack(
                            segms, dim=0).detach().cpu().numpy()
                    else:
                        segms = np.stack(segms, axis=0)

                scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
                bboxes = bboxes[:, 0:4] if bboxes.shape[1] == 5 else bboxes
                assert bboxes.shape[1] == 4

                detection_boxes.append(bboxes)
                detection_scores.append(scores)
                detection_classes.append(labels)
                detection_masks.append(segms)

            assert len(img_metas) == 1
            outputs = {
                'detection_boxes': detection_boxes,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
                'detection_masks': detection_masks,
                'img_metas': img_metas[0]
            }
            return outputs

        setattr(cls, 'forward_test', _new_forward_test)


def update_rpn_head():
    logging.warning('refactor mmdet.models.RPNHead, add `norm_cfg`')
    from mmdet.models.builder import HEADS
    HEADS._module_dict.pop('RPNHead', None)
    from mmdet.models import RPNHead as _RPNHead

    @HEADS.register_module()
    class RPNHead(_RPNHead):
        """RPN head with norm.
        Args:
            in_channels (int): Number of channels in the input feature map.
            init_cfg (dict or list[dict], optional): Initialization config dict.
            num_convs (int): Number of convolution layers in the head. Default 1.
        """  # noqa: W605

        def __init__(self,
                     in_channels,
                     init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                     num_convs=1,
                     norm_cfg=None,
                     **kwargs):

            self.norm_cfg = norm_cfg
            super(RPNHead, self).__init__(
                in_channels, init_cfg=init_cfg, num_convs=num_convs, **kwargs)

        def _init_layers(self):
            """Initialize layers of the head."""
            if self.num_convs > 1:
                rpn_convs = []
                for i in range(self.num_convs):
                    if i == 0:
                        in_channels = self.in_channels
                    else:
                        in_channels = self.feat_channels
                    # use ``inplace=False`` to avoid error: one of the variables
                    # needed for gradient computation has been modified by an
                    # inplace operation.
                    rpn_convs.append(
                        ConvModule(
                            in_channels,
                            self.feat_channels,
                            3,
                            padding=1,
                            norm_cfg=self.norm_cfg,
                            inplace=False))
                self.rpn_conv = nn.Sequential(*rpn_convs)
            else:
                self.rpn_conv = nn.Conv2d(
                    self.in_channels, self.feat_channels, 3, padding=1)
            self.rpn_cls = nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.cls_out_channels, 1)
            self.rpn_reg = nn.Conv2d(self.feat_channels,
                                     self.num_base_priors * 4, 1)


def dynamic_adapt_for_mmlab(cfg):
    mmlab_modules_cfg = cfg.get('mmlab_modules', [])
    if len(mmlab_modules_cfg) > 1:
        adapter = MMAdapter(mmlab_modules_cfg)
        adapter.adapt_mmlab_modules()
