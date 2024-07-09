# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import json
import logging
import pickle
from collections import OrderedDict
from distutils.version import LooseVersion
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as t_f
from mmcv.utils import Config

from easycv.file import io
from easycv.framework.errors import NotImplementedError, ValueError
from easycv.models import (DINO, MOCO, SWAV, YOLOX, BEVFormer, Classification,
                           MoBY, SkeletonGCN, TopDown, build_model)
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.misc import encode_str_to_tensor

__all__ = [
    'export',
    'PreProcess',
    'ModelExportWrapper',
    'ProcessExportWrapper',
]


def export(cfg, ckpt_path, filename, model=None, **kwargs):
    """ export model for inference

    Args:
        cfg: Config object
        ckpt_path (str): path to checkpoint file
        filename (str): filename to save exported models
        model (nn.module): model instance
    """
    if hasattr(cfg.model, 'pretrained'):
        logging.warning(
            'Export needs to set model.pretrained to false to avoid hanging during distributed training'
        )
        cfg.model.pretrained = False

    if model is None:
        model = build_model(cfg.model)

    if ckpt_path != 'dummy':
        load_checkpoint(model, ckpt_path, map_location='cpu')
    else:
        if hasattr(cfg.model, 'backbone') and hasattr(cfg.model.backbone,
                                                      'pretrained'):
            logging.warning(
                'Export needs to set model.backbone.pretrained to false to avoid hanging during distributed training'
            )
            cfg.model.backbone.pretrained = False

    if isinstance(model, MOCO) or isinstance(model, DINO):
        _export_moco(model, cfg, filename, **kwargs)
    elif isinstance(model, MoBY):
        _export_moby(model, cfg, filename, **kwargs)
    elif isinstance(model, SWAV):
        _export_swav(model, cfg, filename, **kwargs)
    elif isinstance(model, Classification):
        _export_cls(model, cfg, filename, **kwargs)
    elif isinstance(model, YOLOX):
        _export_yolox(model, cfg, filename, **kwargs)
    elif isinstance(model, BEVFormer):
        _export_bevformer(model, cfg, filename, **kwargs)
    elif isinstance(model, TopDown):
        _export_pose_topdown(model, cfg, filename, **kwargs)
    elif isinstance(model, SkeletonGCN):
        _export_stgcn(model, cfg, filename, **kwargs)
    elif hasattr(cfg, 'export') and getattr(cfg.export, 'use_jit', False):
        export_jit_model(model, cfg, filename, **kwargs)
        return
    else:
        _export_common(model, cfg, filename, **kwargs)


def _export_common(model, cfg, filename):
    """ export model, add cfg dict to checkpoint['meta']['config'] without process

    Args:
        model (nn.Module):  model to be exported
        cfg: Config object
        filename (str): filename to save exported models
    """
    if not hasattr(cfg, 'test_pipeline'):
        logging.warning('`test_pipeline` not found in export model config!')

    # meta config is type of mmcv.Config, to keep the original config type
    # json will dump int as str
    if isinstance(cfg, Config):
        cfg = cfg._cfg_dict

    meta = dict(config=cfg)
    checkpoint = dict(
        state_dict=model.state_dict(), meta=meta, author='EvTorch')
    with io.open(filename, 'wb') as ofile:
        torch.save(checkpoint, ofile)


def _export_jit_and_blade(model, cfg, filename, dummy_inputs, fp16=False):

    def _trace_model():
        with torch.no_grad():
            if hasattr(model, 'forward_export'):
                model.forward = model.forward_export
            else:
                model.forward = model.forward_test
            trace_model = torch.jit.trace(
                model,
                copy.deepcopy(dummy_inputs),
                strict=False,
                check_trace=False)
        return trace_model

    export_type = cfg.export.get('type')
    if export_type in ['jit', 'blade']:
        if fp16:
            with torch.cuda.amp.autocast():
                trace_model = _trace_model()
        else:
            trace_model = _trace_model()
        torch.jit.save(trace_model, filename + '.jit')
    else:
        raise NotImplementedError(f'Not support export type {export_type}!')

    if export_type == 'jit':
        return

    blade_config = cfg.export.get('blade_config')

    from easycv.toolkit.blade import blade_env_assert, blade_optimize
    assert blade_env_assert()

    def _get_blade_model():
        blade_model = blade_optimize(
            speed_test_model=model,
            model=trace_model,
            inputs=copy.deepcopy(dummy_inputs),
            blade_config=blade_config,
            static_opt=False,
            min_num_nodes=None,
            check_inputs=False,
            fp16=fp16)
        return blade_model

    # optimize model with blade
    if fp16:
        with torch.cuda.amp.autocast():
            blade_model = _get_blade_model()
    else:
        blade_model = _get_blade_model()

    with io.open(filename + '.blade', 'wb') as ofile:
        torch.jit.save(blade_model, ofile)


def _export_onnx_cls(model, model_config, cfg, filename, meta):
    support_backbones = {
        'ResNet': {
            'depth': [50]
        },
        'MobileNetV2': {},
        'Inception3': {},
        'Inception4': {},
        'ResNeXt': {
            'depth': [50]
        }
    }
    if model_config['backbone'].get('type', None) not in support_backbones:
        tmp = ' '.join(support_backbones.keys())
        info_str = f'Only support export onnx model for {tmp} now!'
        raise ValueError(info_str)
    configs = support_backbones[model_config['backbone'].get('type')]
    for k, v in configs.items():
        if v[0].__class__(model_config['backbone'].get(k, None)) not in v:
            raise ValueError(
                f"Unsupport config for {model_config['backbone'].get('type')}")

    # save json config for test_pipline and class
    with io.open(
            filename +
            '.config.json' if filename.endswith('onnx') else filename +
            '.onnx.config.json', 'w') as ofile:
        json.dump(meta, ofile)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    img_size = int(cfg.image_size2)
    x_input = torch.randn((1, 3, img_size, img_size)).to(device)
    torch.onnx.export(
        model,
        (x_input, 'onnx'),
        filename if filename.endswith('onnx') else filename + '.onnx',
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )


def _export_cls(model, cfg, filename):
    """ export cls (cls & metric learning)model and preprocess config

    Args:
        model (nn.Module):  model to be exported
        cfg: Config object
        filename (str): filename to save exported models
    """
    if hasattr(cfg, 'export'):
        export_cfg = cfg.export
    else:
        export_cfg = dict(export_neck=False)

    export_type = export_cfg.get('export_type', 'raw')
    export_neck = export_cfg.get('export_neck', True)
    label_map_path = cfg.get('label_map_path', None)
    class_list = None
    if label_map_path is not None:
        class_list = io.open(label_map_path).readlines()
    elif hasattr(cfg, 'class_list'):
        class_list = cfg.class_list
    elif hasattr(cfg, 'CLASSES'):
        class_list = cfg.CLASSES

    model_config = dict(
        type='Classification',
        backbone=replace_syncbn(cfg.model.backbone),
    )

    # avoid load pretrained model
    model_config['pretrained'] = False

    if export_neck:
        if hasattr(cfg.model, 'neck'):
            model_config['neck'] = cfg.model.neck
        if hasattr(cfg.model, 'head'):
            model_config['head'] = cfg.model.head
    else:
        print("this cls model doesn't contain cls head, we add a dummy head!")
        model_config['head'] = head = dict(
            type='ClsHead',
            with_avg_pool=True,
            in_channels=model_config['backbone'].get('num_classes', 2048),
            num_classes=1000,
        )

    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if hasattr(cfg, 'test_pipeline'):
        test_pipeline = cfg.test_pipeline
        for pipe in test_pipeline:
            if pipe['type'] == 'Collect':
                pipe['keys'] = ['img']
    else:
        test_pipeline = [
            dict(type='Resize', size=[224, 224]),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Collect', keys=['img'])
        ]

    config = dict(
        model=model_config,
        test_pipeline=test_pipeline,
        class_list=class_list,
    )

    meta = dict(config=json.dumps(config))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if k.startswith('backbone'):
            state_dict[k] = v
        if export_neck and (k.startswith('neck') or k.startswith('head')):
            state_dict[k] = v

    if export_type == 'raw':
        checkpoint = dict(state_dict=state_dict, meta=meta, author='EasyCV')
        with io.open(filename, 'wb') as ofile:
            torch.save(checkpoint, ofile)
    elif export_type == 'onnx':
        _export_onnx_cls(model, model_config, cfg, filename, config)
    else:
        raise ValueError('Only support export onnx/raw model!')


def _export_yolox(model, cfg, filename):
    """ export cls (cls & metric learning)model and preprocess config
    Args:
        model (nn.Module):  model to be exported
        cfg: Config object
        filename (str): filename to save exported models
    """

    if hasattr(cfg, 'export'):
        export_type = getattr(cfg.export, 'export_type', 'raw')
        default_export_type_list = ['raw', 'jit', 'blade', 'onnx']
        if export_type not in default_export_type_list:
            logging.warning(
                'YOLOX-PAI only supports the export type as  [raw,jit,blade,onnx], otherwise we use raw as default'
            )
            export_type = 'raw'

        model.export_type = export_type

        if export_type != 'raw':
            from easycv.utils.misc import reparameterize_models
            # only when we use jit or blade, we need to reparameterize_models before export
            model = reparameterize_models(model)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = copy.deepcopy(model)

            preprocess_jit = cfg.export.get('preprocess_jit', False)

            batch_size = cfg.export.get('batch_size', 1)
            static_opt = cfg.export.get('static_opt', True)
            use_trt_efficientnms = cfg.export.get('use_trt_efficientnms',
                                                  False)
            # assert image scale and assgin input
            img_scale = cfg.get('img_scale', (640, 640))

            assert (
                len(img_scale) == 2
            ), 'Export YoloX predictor config contains img_scale must be (int, int) tuple!'

            input = 255 * torch.rand((batch_size, 3) + tuple(img_scale))

            # assert use_trt_efficientnms only happens when static_opt=True
            if static_opt is not True:
                assert (
                    use_trt_efficientnms == False
                ), 'Export YoloX predictor use_trt_efficientnms=True only when use static_opt=True!'

            # allow to save a preprocess jit model with exported model
            save_preprocess_jit = False

            if preprocess_jit:
                save_preprocess_jit = True

            # set model use_trt_efficientnms
            if use_trt_efficientnms:
                from easycv.toolkit.blade import create_tensorrt_efficientnms
                if hasattr(model, 'get_nmsboxes_num'):
                    nmsbox_num = int(model.get_nmsboxes_num(img_scale))
                else:
                    logging.warning(
                        'PAI-YOLOX: use_trt_efficientnms encounter model has no attr named get_nmsboxes_num, use 8400 (80*80+40*40+20*20)cas default!'
                    )
                    nmsbox_num = 8400

                tmp_example_scores = torch.randn(
                    [batch_size, nmsbox_num, 4 + 1 + len(cfg.CLASSES)],
                    dtype=torch.float32)
                logging.warning(
                    'PAI-YOLOX: use_trt_efficientnms with staic shape [{}, {}, {}]'
                    .format(batch_size, nmsbox_num, 4 + 1 + len(cfg.CLASSES)))
                model.trt_efficientnms = create_tensorrt_efficientnms(
                    tmp_example_scores,
                    iou_thres=model.nms_thre,
                    score_thres=model.test_conf)
                model.use_trt_efficientnms = True

            model.eval()
            model.to(device)

            model_export = ModelExportWrapper(
                model,
                input.to(device),
                trace_model=True,
            )

            model_export.eval().to(device)

            # trace model
            yolox_trace = torch.jit.trace(model_export, input.to(device))

            # save export model
            if export_type == 'blade':
                blade_config = cfg.export.get(
                    'blade_config',
                    dict(enable_fp16=True, fp16_fallback_op_ratio=0.3))

                from easycv.toolkit.blade import blade_env_assert, blade_optimize
                assert blade_env_assert()

                # optimize model with blade
                yolox_blade = blade_optimize(
                    speed_test_model=model,
                    model=yolox_trace,
                    inputs=(input.to(device), ),
                    blade_config=blade_config,
                    static_opt=static_opt)

                with io.open(filename + '.blade', 'wb') as ofile:
                    torch.jit.save(yolox_blade, ofile)
                with io.open(filename + '.blade.config.json', 'w') as ofile:
                    config = dict(
                        model=cfg.model,
                        export=cfg.export,
                        test_pipeline=cfg.test_pipeline,
                        classes=cfg.CLASSES)

                    json.dump(config, ofile)

            if export_type == 'onnx':

                with io.open(
                        filename + '.config.json' if filename.endswith('onnx')
                        else filename + '.onnx.config.json', 'w') as ofile:
                    config = dict(
                        model=cfg.model,
                        export=cfg.export,
                        test_pipeline=cfg.test_pipeline,
                        classes=cfg.CLASSES)

                    json.dump(config, ofile)

                torch.onnx.export(
                    model,
                    input.to(device),
                    filename if filename.endswith('onnx') else filename +
                    '.onnx',
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                )

            if export_type == 'jit':
                with io.open(filename + '.jit', 'wb') as ofile:
                    torch.jit.save(yolox_trace, ofile)

                with io.open(filename + '.jit.config.json', 'w') as ofile:
                    config = dict(
                        model=cfg.model,
                        export=cfg.export,
                        test_pipeline=cfg.test_pipeline,
                        classes=cfg.CLASSES)

                    json.dump(config, ofile)

            # save export preprocess/postprocess
            if save_preprocess_jit:
                tpre_input = 255 * torch.rand((batch_size, ) + img_scale +
                                              (3, ))
                tpre = ProcessExportWrapper(
                    example_inputs=tpre_input.to(device),
                    process_fn=PreProcess(
                        target_size=img_scale, keep_ratio=True))
                tpre.eval().to(device)

                preprocess = torch.jit.script(tpre)
                with io.open(filename + '.preprocess', 'wb') as prefile:
                    torch.jit.save(preprocess, prefile)

        else:
            if hasattr(cfg, 'test_pipeline'):
                # with last pipeline Collect
                test_pipeline = cfg.test_pipeline
                print(test_pipeline)
            else:
                print('test_pipeline not found, using default preprocessing!')
                raise ValueError('export model config without test_pipeline')

            config = dict(
                model=cfg.model,
                test_pipeline=test_pipeline,
                CLASSES=cfg.CLASSES,
            )

            meta = dict(config=json.dumps(config))
            checkpoint = dict(
                state_dict=model.state_dict(), meta=meta, author='EasyCV')
            with io.open(filename, 'wb') as ofile:
                torch.save(checkpoint, ofile)


def _export_swav(model, cfg, filename):
    """ export cls (cls & metric learning)model and preprocess config

    Args:
        model (nn.Module):  model to be exported
        cfg: Config object
        filename (str): filename to save exported models
    """
    if hasattr(cfg, 'export'):
        export_cfg = cfg.export
    else:
        export_cfg = dict(export_neck=False)
    export_neck = export_cfg.get('export_neck', False)

    tbackbone = replace_syncbn(cfg.model.backbone)

    model_config = dict(
        type='Classification',
        pretrained=False,  # avoid loading default pretrained backbone model
        backbone=tbackbone,
    )

    if export_neck and hasattr(cfg.model, 'neck'):
        cfg.model.neck.export = True
        cfg.model.neck.with_avg_pool = True
        model_config['neck'] = cfg.model.neck

    if hasattr(model_config, 'neck'):
        output_channels = model_config['neck']['out_channels']
    else:
        output_channels = 2048

    model_config['head'] = head = dict(
        type='ClsHead',
        with_avg_pool=False,
        in_channels=output_channels,
        num_classes=1000,
    )

    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if hasattr(cfg, 'test_pipeline'):
        test_pipeline = cfg.test_pipeline
    else:
        test_pipeline = [
            dict(type='Resize', size=[224, 224]),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]

    config = dict(model=model_config, test_pipeline=test_pipeline)
    meta = dict(config=json.dumps(config))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if k.startswith('backbone'):
            state_dict[k] = v
        elif k.startswith('head'):
            state_dict[k] = v
        # feature extractor need classification model, classification mode = extract only support neck_0 to infer after sprint2101
        # swav's neck is saved as 'neck.'
        elif export_neck and (k.startswith('neck.')):
            new_key = k.replace('neck.', 'neck_0.')
            state_dict[new_key] = v

    checkpoint = dict(state_dict=state_dict, meta=meta, author='EasyCV')
    with io.open(filename, 'wb') as ofile:
        torch.save(checkpoint, ofile)


def _export_moco(model, cfg, filename):
    """ export model and preprocess config

    Args:
        model (nn.Module):  model to be exported
        cfg: Config object
        filename (str): filename to save exported models
    """
    if hasattr(cfg, 'export'):
        export_cfg = cfg.export
    else:
        export_cfg = dict(export_neck=False)
    export_neck = export_cfg.get('export_neck', False)

    model_config = dict(
        type='Classification',
        pretrained=False,  # avoid loading default pretrained backbone model
        backbone=replace_syncbn(cfg.model.backbone),
        head=dict(
            type='ClsHead',
            with_avg_pool=True,
            in_channels=2048,
            num_classes=1000,
        ),
    )
    if export_neck:
        model_config['neck'] = cfg.model.neck

    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_pipeline = [
        dict(type='Resize', size=[224, 224]),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
    ]

    config = dict(
        model=model_config,
        test_pipeline=test_pipeline,
    )

    meta = dict(config=json.dumps(config))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if k.startswith('backbone'):
            state_dict[k] = v
        neck_key = 'encoder_q.1'
        if export_neck and k.startswith(neck_key):
            new_key = k.replace(neck_key, 'neck_0')
            state_dict[new_key] = v

    checkpoint = dict(state_dict=state_dict, meta=meta, author='EasyCV')
    with io.open(filename, 'wb') as ofile:
        torch.save(checkpoint, ofile)


def _export_moby(model, cfg, filename):
    """ export model and preprocess config

    Args:
        model (nn.Module):  model to be exported
        cfg: Config object
        filename (str): filename to save exported models
    """
    if hasattr(cfg, 'export'):
        export_cfg = cfg.export
    else:
        export_cfg = dict(export_neck=False)
    export_neck = export_cfg.get('export_neck', False)

    model_config = dict(
        type='Classification',
        pretrained=False,  # avoid loading default pretrained backbone model
        backbone=replace_syncbn(cfg.model.backbone),
        head=dict(
            type='ClsHead',
            with_avg_pool=True,
            in_channels=2048,
            num_classes=1000,
        ),
    )
    if export_neck:
        model_config['neck'] = cfg.model.neck

    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_pipeline = [
        dict(type='Resize', size=[224, 224]),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
    ]

    config = dict(
        model=model_config,
        test_pipeline=test_pipeline,
    )

    meta = dict(config=json.dumps(config))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if k.startswith('backbone'):
            state_dict[k] = v
        neck_key = 'projector_q'
        if export_neck and k.startswith(neck_key):
            new_key = k.replace(neck_key, 'neck_0')
            state_dict[new_key] = v

    checkpoint = dict(state_dict=state_dict, meta=meta, author='EasyCV')
    with io.open(filename, 'wb') as ofile:
        torch.save(checkpoint, ofile)


def export_jit_model(model, cfg, filename):
    """ export jit model

    Args:
        model (nn.Module):  model to be exported
        cfg: Config object
        filename (str): filename to save exported models
    """
    model_jit = torch.jit.script(model)
    with io.open(filename, 'wb') as ofile:
        torch.jit.save(model_jit, ofile)


def _export_bevformer(model, cfg, filename, fp16=False, dummy_inputs=None):
    if not cfg.adapt_jit:
        raise ValueError(
            '"cfg.adapt_jit" must be True when export jit trace or blade model.'
        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = copy.deepcopy(model)
    model.eval()
    model.to(device)

    def _dummy_inputs():
        # dummy inputs
        bacth_size, queue_len, cams_num = 1, 1, 6
        img_size = (928, 1600)
        img = torch.rand([cams_num, 3, img_size[0], img_size[1]]).to(device)
        can_bus = torch.rand([18]).to(device)
        lidar2img = torch.rand([6, 4, 4]).to(device)
        img_shape = torch.tensor([[img_size[0], img_size[1], 3]] *
                                 cams_num).to(device)
        dummy_scene_token = 'dummy_scene_token'
        scene_token = encode_str_to_tensor(dummy_scene_token).to(device)
        prev_scene_token = scene_token
        prev_bev = torch.rand([cfg.bev_h * cfg.bev_w, 1,
                               cfg.embed_dim]).to(device)
        prev_pos = torch.tensor(0)
        prev_angle = torch.tensor(0)
        img_metas = {
            'can_bus': can_bus,
            'lidar2img': lidar2img,
            'img_shape': img_shape,
            'scene_token': scene_token,
            'prev_bev': prev_bev,
            'prev_pos': prev_pos,
            'prev_angle': prev_angle,
            'prev_scene_token': prev_scene_token
        }
        return img, img_metas

    if dummy_inputs is None:
        dummy_inputs = _dummy_inputs()

    _export_jit_and_blade(model, cfg, filename, dummy_inputs, fp16=fp16)


def _export_pose_topdown(model, cfg, filename, fp16=False, dummy_inputs=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = copy.deepcopy(model)
    model.eval()
    model.to(device)

    if hasattr(cfg, 'export') and getattr(cfg.export, 'type', 'raw') == 'raw':
        from mmcv.utils.path import is_filepath

        if hasattr(cfg, 'dataset_info') and is_filepath(cfg.dataset_info):
            dataset_info_cfg = Config.fromfile(cfg.dataset_info)
            cfg.dataset_info = dataset_info_cfg._cfg_dict['dataset_info']

        return _export_common(model, cfg, filename)

    def _dummy_inputs(cfg):
        from easycv.datasets.pose.data_sources.top_down import DatasetInfo
        from easycv.datasets.pose.data_sources.wholebody.wholebody_coco_source import WHOLEBODY_COCO_DATASET_INFO
        from easycv.datasets.pose.data_sources.hand.coco_hand import COCO_WHOLEBODY_HAND_DATASET_INFO
        from easycv.datasets.pose.data_sources.coco import COCO_DATASET_INFO

        data_type = cfg.data.train.data_source.type
        data_info_map = {
            'WholeBodyCocoTopDownSource': WHOLEBODY_COCO_DATASET_INFO,
            'PoseTopDownSourceCoco': COCO_DATASET_INFO,
            'HandCocoPoseTopDownSource': COCO_WHOLEBODY_HAND_DATASET_INFO
        }
        dataset_info = DatasetInfo(data_info_map[data_type])
        flip_pairs = dataset_info.flip_pairs
        image_size = cfg.data_cfg.image_size
        img = torch.rand([1, 3, image_size[1], image_size[0]]).to(device)
        img_metas = [{
            'image_id': torch.tensor(0),
            'center': torch.tensor([426., 451.]),
            'scale': torch.tensor([4., 5.]),
            'rotation': torch.tensor(0),
            'flip_pairs': torch.tensor(flip_pairs),
            'bbox_id': torch.tensor(0)
        }]
        return img, img_metas

    if dummy_inputs is None:
        dummy_inputs = _dummy_inputs(cfg)

    _export_jit_and_blade(model, cfg, filename, dummy_inputs, fp16=fp16)


def _export_stgcn(model, cfg, filename, fp16=False, dummy_inputs=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = copy.deepcopy(model)
    model.eval()
    model.to(device)

    if hasattr(cfg, 'export') and getattr(cfg.export, 'type', 'raw') == 'raw':
        return _export_common(model, cfg, filename)

    def _dummy_inputs(device):
        keypoints = torch.randn([1, 3, 300, 17, 2]).to(device)
        return (keypoints, )

    if dummy_inputs is None:
        dummy_inputs = _dummy_inputs(device)

    _export_jit_and_blade(model, cfg, filename, dummy_inputs, fp16=fp16)


def replace_syncbn(backbone_cfg):
    if 'norm_cfg' in backbone_cfg.keys():
        if backbone_cfg['norm_cfg']['type'] == 'SyncBN':
            backbone_cfg['norm_cfg']['type'] = 'BN'
        elif backbone_cfg['norm_cfg']['type'] == 'SyncIBN':
            backbone_cfg['norm_cfg']['type'] = 'IBN'

    return backbone_cfg


if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):

    @torch.jit.script
    class PreProcess:
        """Process the data input to model.

        Args:
            target_size (Tuple[int, int]): output spatial size.
            keep_ratio (bool): Whether to keep the aspect ratio when resizing the image.
        """

        def __init__(self,
                     target_size: Tuple[int, int] = (640, 640),
                     keep_ratio: bool = True):

            self.target_size = target_size
            self.keep_ratio = keep_ratio

        def __call__(
            self, image: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict[str, Tuple[float, float]]]:
            """
            Args:
                image (torch.Tensor): image format should be [b, H, W, C]
            """
            input_h, input_w = self.target_size
            image = image.permute(0, 3, 1, 2)

            # rgb2bgr
            image = image[:, torch.tensor([2, 1, 0]), :, :]

            ori_h, ori_w = image.shape[-2:]

            mean = [123.675, 116.28, 103.53]
            std = [58.395, 57.12, 57.375]

            if not self.keep_ratio:
                out_image = t_f.resize(image, [input_h, input_w])
                out_image = t_f.normalize(out_image, mean, std)
                pad_l, pad_t, scale = 0, 0, 1.0
            else:
                scale = min(input_h / ori_h, input_w / ori_w)
                resize_h, resize_w = int(ori_h * scale), int(ori_w * scale)

                # pay attention to the padding position! In mmcv, padding is conducted in the right and bottom
                pad_h, pad_w = input_h - resize_h, input_w - resize_w
                pad_l, pad_t = 0, 0
                pad_r, pad_b = pad_w - pad_l, pad_h - pad_t
                out_image = t_f.resize(image, [resize_h, resize_w])
                out_image = t_f.pad(
                    out_image, [pad_l, pad_t, pad_r, pad_b], fill=114)

                # float is necessary to match the preprocess result with mmcv
                out_image = out_image.float()

                out_image = t_f.normalize(out_image, mean, std)

            h, w = out_image.shape[-2:]
            output_info = {
                'pad': (float(pad_l), float(pad_t)),
                'scale_factor': (float(scale), float(scale)),
                'ori_img_shape': (float(ori_h), float(ori_w)),
                'img_shape': (float(h), float(w))
            }

            return out_image, output_info

else:
    PreProcess = None


class ModelExportWrapper(torch.nn.Module):

    def __init__(self,
                 model,
                 example_inputs,
                 trace_model: bool = True) -> None:
        super().__init__()

        self.model = model
        if hasattr(self.model, 'export_init'):
            self.model.export_init()

        self.example_inputs = example_inputs

        self.trace_model = trace_model

        if self.trace_model:
            try:
                self.trace_module()
            except RuntimeError:
                # well trained model will generate reasonable result, otherwise, we should change model.test_conf=0.0 to avoid tensor in inference to be empty
                logging.warning(
                    'PAI-YOLOX: set model.test_conf=0.0 to avoid tensor in inference to be empty'
                )
                model.test_conf = 0.0
                self.trace_module()

    def trace_module(self, **kwargs):
        trace_model = torch.jit.trace_module(
            self.model, {'forward_export': self.example_inputs}, **kwargs)
        self.model = trace_model

    def forward(self, image):

        with torch.no_grad():
            model_output = self.model.forward_export(image)

        return model_output


class ProcessExportWrapper(torch.nn.Module):
    """
        split the preprocess that can be wrapped as a preprocess jit model
        the preproprocess procedure cannot be optimized in an end2end blade model due to dynamic shape problem
    """

    def __init__(self,
                 example_inputs,
                 process_fn: Optional[Callable] = None) -> None:
        super().__init__()
        self.process_fn = process_fn

    def forward(self, image):
        with torch.no_grad():
            output = self.process_fn(image)

        return output
