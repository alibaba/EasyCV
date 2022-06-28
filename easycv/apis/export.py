# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import json
import logging
from collections import OrderedDict
from distutils.version import LooseVersion
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as t_f
from mmcv.utils import Config

from easycv.file import io
from easycv.models import (DINO, MOCO, SWAV, YOLOX, Classification, MoBY,
                           build_model)
from easycv.utils.bbox_util import scale_coords
from easycv.utils.checkpoint import load_checkpoint

__all__ = [
    'export', 'PreProcess', 'DetPostProcess', 'End2endModelExportWrapper'
]


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

    if hasattr(cfg, 'export') and (getattr(cfg.export, 'use_jit', False) or
                                   getattr(cfg.export, 'export_blade', False)):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = copy.deepcopy(model)
        model.eval()
        model.to(device)

        end2end = cfg.export.get('end2end', False)
        if LooseVersion(torch.__version__) < LooseVersion('1.7.0') and end2end:
            raise ValueError('`end2end` only support torch1.7.0 and later!')

        batch_size = cfg.export.get('batch_size', 1)
        img_scale = cfg.get('img_scale', (640, 640))
        assert (
            len(img_scale) == 2
        ), 'Export YoloX predictor config contains img_scale must be (int, int) tuple!'

        input = 255 * torch.rand((batch_size, 3) + img_scale)

        model_export = End2endModelExportWrapper(
            model,
            input.to(device),
            preprocess_fn=PreProcess(target_size=img_scale, keep_ratio=True)
            if end2end else None,
            postprocess_fn=DetPostProcess(max_det=100, score_thresh=0.5)
            if end2end else None,
            trace_model=True,
        )

        model_export.eval().to(device)

        # well trained model will generate reasonable result, otherwise, we should change model.test_conf=0.0 to avoid tensor in inference to be empty
        # use trace is a litter bit faster than script. But it is not supported in an end2end model.
        if end2end:
            yolox_trace = torch.jit.script(model_export)

        else:
            yolox_trace = torch.jit.trace(model_export, input.to(device))

        if getattr(cfg.export, 'export_blade', False):
            blade_config = cfg.export.get('blade_config',
                                          dict(enable_fp16=True))

            from easycv.toolkit.blade import blade_env_assert, blade_optimize

            assert blade_env_assert()

            if end2end:
                input = 255 * torch.rand(img_scale + (3, ))

            yolox_blade = blade_optimize(
                script_model=model,
                model=yolox_trace,
                inputs=(input.to(device), ),
                blade_config=blade_config)

            with io.open(filename + '.blade', 'wb') as ofile:
                torch.jit.save(yolox_blade, ofile)
            with io.open(filename + '.blade.config.json', 'w') as ofile:
                config = dict(
                    export=cfg.export,
                    test_pipeline=cfg.test_pipeline,
                    classes=cfg.CLASSES)

                json.dump(config, ofile)

        if getattr(cfg.export, 'use_jit', False):
            with io.open(filename + '.jit', 'wb') as ofile:
                torch.jit.save(yolox_trace, ofile)

            with io.open(filename + '.jit.config.json', 'w') as ofile:
                config = dict(
                    export=cfg.export,
                    test_pipeline=cfg.test_pipeline,
                    classes=cfg.CLASSES)

                json.dump(config, ofile)

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
                image (torch.Tensor): image format should be [H, W, C]
            """
            input_h, input_w = self.target_size
            image = image.permute(2, 0, 1)

            # rgb2bgr
            image = image[[2, 1, 0], :, :]

            image = torch.unsqueeze(image, 0)
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

    @torch.jit.script
    class DetPostProcess:
        """Process output values of detection models.

        Args:
            max_det: max number of detections to keep.
        """

        def __init__(self, max_det: int = 100, score_thresh: float = 0.5):
            self.max_det = max_det
            self.score_thresh = score_thresh

        def __call__(
            self, output: List[torch.Tensor], sample_info: Dict[str,
                                                                Tuple[float,
                                                                      float]]
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                output (List[torch.Tensor]): model output
                sample_info (dict): sample infomation containing keys:
                    pad: Pixel size of each side width and height padding
                    scale_factor: the preprocessing scale factor
                    ori_img_shape: original image shape
                    img_shape: processed image shape
            """
            pad = sample_info['pad']
            scale_factor = sample_info['scale_factor']
            ori_h, ori_w = sample_info['ori_img_shape']
            h, w = sample_info['img_shape']

            output = output[0]

            det_out = output[:self.max_det]

            det_out = scale_coords((int(h), int(w)), det_out,
                                   (int(ori_h), int(ori_w)),
                                   (scale_factor, pad))

            detection_boxes = det_out[:, :4].cpu()
            detection_scores = (det_out[:, 4] * det_out[:, 5]).cpu()
            detection_classes = det_out[:, 6].cpu().int()

            out = {
                'detection_boxes': detection_boxes,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
            }

            return out
else:
    PreProcess = None
    DetPostProcess = None


class End2endModelExportWrapper(torch.nn.Module):
    """Model export wrapper that supports end-to-end export of pre-processing and post-processing.
    We support some built-in preprocessing and postprocessing functions.
    If the requirements are not met, you can customize the preprocessing and postprocessing functions.
    The custom functions must support satisfy requirements of `torch.jit.script`,
    please refer to: https://pytorch.org/docs/stable/jit_language_reference_v2.html

    Args:
        model (torch.nn.Module):  `torch.nn.Module` that will be run with `example_inputs`.
            `model` arguments and return values must be tensors or (possibly nested) tuples
            that contain tensors. When a module is passed `torch.jit.trace`, only the
            ``forward_export`` method is run and traced (see :func:`torch.jit.trace
            <torch.jit.trace_module>` for details).
        example_inputs (tuple or torch.Tensor):  A tuple of example inputs that
            will be passed to the function while tracing. The resulting trace
            can be run with inputs of different types and shapes assuming the
            traced operations support those types and shapes. `example_inputs`
            may also be a single Tensor in which case it is automatically
            wrapped in a tuple.
        preprocess_fn (callable or None): A Python function for processing example_input.
            If there is only one return value, it will be passed to `model.forward_export`.
            If there are multiple return values, the first return value will be passed to `model.forward_export`,
            and the remaining return values ​​will be passed to `postprocess_fn`.
        postprocess_fn (callable or None): A Python function for processing the output value of the model.
            If `preprocess_fn` has multiple outputs, the output value of `preprocess_fn`
            will also be passed to `postprocess_fn`. For details, please refer to: `preprocess_fn`.
        trace_model (bool): If True, before exporting the end-to-end model,
            `torch.jit.trace` will be used to export the `model` first.
            Traceing an ``nn.Module`` by default will compile the ``forward_export`` method and recursively.

    Examples:
        import torch

        batch_size = 1
        example_inputs = 255 * torch.rand((batch_size, 3, 640, 640), device='cuda')
        end2end_model = End2endModelExportWrapper(
            model,
            example_inputs,
            preprocess_fn=PreProcess(target_size=(640, 640)),  # `PreProcess` refer to ev_torch.apis.export.PreProcess
            postprocess_fn=DetPostProcess()  # `DetPostProcess` refer to ev_torch.apis.export.DetPostProcess
            trace_model=True)

        model_script = torch.jit.script(end2end_model)
        with io.open('/tmp/model.jit', 'wb') as f:
            torch.jit.save(model_script, f)
    """

    def __init__(self,
                 model,
                 example_inputs,
                 preprocess_fn: Optional[Callable] = None,
                 postprocess_fn: Optional[Callable] = None,
                 trace_model: bool = True) -> None:
        super().__init__()

        self.model = model
        if hasattr(self.model, 'export_init'):
            self.model.export_init()

        self.example_inputs = example_inputs
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn
        self.trace_model = trace_model
        if self.trace_model:
            self.trace_module()

    def trace_module(self, **kwargs):
        trace_model = torch.jit.trace_module(
            self.model, {'forward_export': self.example_inputs}, **kwargs)
        self.model = trace_model

    def forward(self, image):
        preprocess_outputs = ()

        with torch.no_grad():
            if self.preprocess_fn is not None:
                output = self.preprocess_fn(image)
                # if multi values ​​are returned, the first one must be image, others ​​are optional,
                # and others will all be passed into postprocess_fn
                if isinstance(output, tuple):
                    image = output[0]
                    preprocess_outputs = output[1:]
                else:
                    image = output

            model_output = self.model.forward_export(image)

            if self.postprocess_fn is not None:
                model_output = self.postprocess_fn(model_output,
                                                   *preprocess_outputs)

        return model_output
