# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
from collections import OrderedDict

import torch
from mmcv.utils import Config

from easycv.file import io
from easycv.models import (DINO, MOCO, SWAV, YOLOX, Classification, MoBY,
                           build_model)
from easycv.utils.checkpoint import load_checkpoint

__all__ = ['export']


def export(cfg, ckpt_path, filename):
    """ export model for inference

    Args:
        cfg: Config object
        ckpt_path (str): path to checkpoint file
        filename (str): filename to save exported models
    """
    model = build_model(cfg.model)
    if ckpt_path != 'dummy':
        load_checkpoint(model, ckpt_path, map_location='cpu')
    else:
        cfg.model.backbone.pretrained = False

    if isinstance(model, MOCO) or isinstance(model, DINO):
        _export_moco(model, cfg, filename)
    elif isinstance(model, MoBY):
        _export_moby(model, cfg, filename)
    elif isinstance(model, SWAV):
        _export_swav(model, cfg, filename)
    elif isinstance(model, Classification):
        _export_cls(model, cfg, filename)
    elif isinstance(model, YOLOX):
        _export_yolox(model, cfg, filename)
    elif hasattr(cfg, 'export') and getattr(cfg.export, 'use_jit', False):
        export_jit_model(model, cfg, filename)
        return
    else:
        _export_common(model, cfg, filename)


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

    export_neck = export_cfg.get('export_neck', True)
    label_map_path = cfg.get('label_map_path', None)
    class_list = None
    if label_map_path is not None:
        class_list = io.open(label_map_path).readlines()
    elif hasattr(cfg, 'class_list'):
        class_list = cfg.class_list

    model_config = dict(
        type='Classification',
        backbone=replace_syncbn(cfg.model.backbone),
    )

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

    checkpoint = dict(state_dict=state_dict, meta=meta, author='EasyCV')
    with io.open(filename, 'wb') as ofile:
        torch.save(checkpoint, ofile)


def _export_yolox(model, cfg, filename):
    """ export cls (cls & metric learning)model and preprocess config

    Args:
        model (nn.Module):  model to be exported
        cfg: Config object
        filename (str): filename to save exported models
    """
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


def replace_syncbn(backbone_cfg):
    if 'norm_cfg' in backbone_cfg.keys():
        if backbone_cfg['norm_cfg']['type'] == 'SyncBN':
            backbone_cfg['norm_cfg']['type'] = 'BN'
        elif backbone_cfg['norm_cfg']['type'] == 'SyncIBN':
            backbone_cfg['norm_cfg']['type'] = 'IBN'

    return backbone_cfg
