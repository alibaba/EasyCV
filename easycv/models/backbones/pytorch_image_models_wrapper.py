# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib
from distutils.version import LooseVersion

import timm
import torch
import torch.nn as nn
from timm.models.helpers import load_pretrained
from timm.models.hub import download_cached_file

from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger
from ..modelzoo import timm_models as model_urls
from ..registry import BACKBONES
from .shuffle_transformer import (shuffletrans_base_p4_w7_224,
                                  shuffletrans_small_p4_w7_224,
                                  shuffletrans_tiny_p4_w7_224)
from .swin_transformer_dynamic import (dynamic_swin_base_p4_w7_224,
                                       dynamic_swin_small_p4_w7_224,
                                       dynamic_swin_tiny_p4_w7_224)
from .vit_transfomer_dynamic import (dynamic_vit_base_p16,
                                     dynamic_vit_huge_p14,
                                     dynamic_vit_large_p16)
from .xcit_transformer import (xcit_large_24_p8, xcit_medium_24_p8,
                               xcit_medium_24_p16, xcit_small_12_p8,
                               xcit_small_12_p16)

_MODEL_MAP = {
    # shuffle_transformer
    'shuffletrans_tiny_p4_w7_224': shuffletrans_tiny_p4_w7_224,
    'shuffletrans_base_p4_w7_224': shuffletrans_base_p4_w7_224,
    'shuffletrans_small_p4_w7_224': shuffletrans_small_p4_w7_224,

    # swin_transformer_dynamic
    'dynamic_swin_tiny_p4_w7_224': dynamic_swin_tiny_p4_w7_224,
    'dynamic_swin_small_p4_w7_224': dynamic_swin_small_p4_w7_224,
    'dynamic_swin_base_p4_w7_224': dynamic_swin_base_p4_w7_224,

    # vit_transfomer_dynamic
    'dynamic_vit_base_p16': dynamic_vit_base_p16,
    'dynamic_vit_large_p16': dynamic_vit_large_p16,
    'dynamic_vit_huge_p14': dynamic_vit_huge_p14,

    # xcit_transformer
    'xcit_small_12_p16': xcit_small_12_p16,
    'xcit_small_12_p8': xcit_small_12_p8,
    'xcit_medium_24_p16': xcit_medium_24_p16,
    'xcit_medium_24_p8': xcit_medium_24_p8,
    'xcit_large_24_p8': xcit_large_24_p8
}


@BACKBONES.register_module
class PytorchImageModelWrapper(nn.Module):
    """Support Backbones From pytorch-image-models.

    The PyTorch community has lots of awesome contributions for image models. PyTorch Image Models (timm) is
    a collection of image models, aim to pull together a wide variety of SOTA models with ability to reproduce
    ImageNet training results.

    Model pages can be found at https://rwightman.github.io/pytorch-image-models/models/

    References: https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self,
                 model_name='resnet50',
                 pretrained=False,
                 checkpoint_path=None,
                 scriptable=None,
                 exportable=None,
                 no_jit=None,
                 **kwargs):
        """
        Inits PytorchImageModelWrapper by timm.create_models
        Args:
            model_name (str): name of model to instantiate
            pretrained (bool): load pretrained ImageNet-1k weights if true
            checkpoint_path (str): path of checkpoint to load after model is initialized
            scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
            exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
            no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)
        """
        super(PytorchImageModelWrapper, self).__init__()

        timm_model_names = timm.list_models(pretrained=False)
        assert model_name in timm_model_names or model_name in _MODEL_MAP, \
            f'{model_name} is not in model_list of timm/fair, please check the model_name!'

        # Default to use backbone without head from timm
        if 'num_classes' not in kwargs:
            kwargs['num_classes'] = 0

        if model_name in model_urls:
            pretrained_path = checkpoint_path or model_urls[model_name]
        else:
            pretrained_path = checkpoint_path
        print('load pretrained from {}'.format(pretrained_path))
        # create model by timm
        if model_name in timm_model_names:
            self.model = timm.create_model(model_name, False, '', scriptable,
                                           exportable, no_jit, **kwargs)

            if pretrained:
                if pretrained_path.endswith('.npz'):
                    pretrained_loc = download_cached_file(
                        pretrained_path, check_hash=False, progress=False)
                    return self.model.load_pretrained(pretrained_loc)
                else:
                    backbone_module = importlib.import_module(
                        self.model.__module__)
                    return load_pretrained(
                        self.model,
                        default_cfg={'url': pretrained_path},
                        filter_fn=backbone_module.checkpoint_filter_fn
                        if hasattr(backbone_module, 'checkpoint_filter_fn')
                        else None)
        elif model_name in _MODEL_MAP:
            # facebook model wrapper
            self.model = _MODEL_MAP[model_name](**kwargs)
            if pretrained:
                if model_name in model_urls.keys():
                    try_max = 3
                    try_idx = 0
                    while try_idx < try_max:
                        try:
                            state_dict = torch.hub.load_state_dict_from_url(
                                url=model_urls[model_name],
                                map_location='cpu',
                            )
                            try_idx += try_max
                        except Exception:
                            try_idx += 1
                            state_dict = {}
                            if try_idx == try_max:
                                print(
                                    'load from url failed ! oh my DLC & OSS, you boys really good! ',
                                    model_urls[model_name])

                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    for key, value in state_dict.items():
                        print(key, ':', value.size())
                    for name, parameters in self.model.named_parameters():
                        print(name, ':', parameters.size())
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    print('%s not in evtorch modelzoo!' % model_name)
        else:
            print(
                f'Error: Fail to create {model_name} with (pretrained={pretrained}, checkpoint_path={checkpoint_path} ...)'
            )

    def forward(self, x):

        o = self.model(x)
        if type(o) == tuple or type(o) == list:
            features = []
            for feature in o:
                while feature.dim() < 4:
                    feature = feature.unsqueeze(-1)
                features.append(feature)
        else:
            while o.dim() < 4:
                o = o.unsqueeze(-1)
            features = [o]

        return tuple(features)
