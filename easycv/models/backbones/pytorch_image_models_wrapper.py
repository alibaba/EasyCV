# Copyright (c) Alibaba, Inc. and its affiliates.
from distutils.version import LooseVersion

import timm
import torch
import torch.nn as nn

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
from .vit_transfomer_dynamic import (dynamic_deit_small_p16,
                                     dynamic_deit_tiny_p16,
                                     dynamic_vit_base_p16,
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
    'dynamic_deit_small_p16': dynamic_deit_small_p16,
    'dynamic_deit_tiny_p16': dynamic_deit_tiny_p16,
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

        # create model by timm
        if model_name in timm_model_names:
            try:
                if pretrained and (model_name in model_urls):
                    self.model = timm.create_model(model_name, False, '',
                                                   scriptable, exportable,
                                                   no_jit, **kwargs)
                    self.init_weights(model_urls[model_name])
                    print('Info: Load model from %s' % model_urls[model_name])

                    if checkpoint_path is not None:
                        self.init_weights(checkpoint_path)
                else:
                    # load from timm
                    if pretrained and model_name.startswith('swin_') and (
                            LooseVersion(
                                torch.__version__) <= LooseVersion('1.6.0')):
                        print(
                            'Warning: Pretrained SwinTransformer from timm may be zipfile extract'
                            ' error while torch<=1.6.0')
                    self.model = timm.create_model(model_name, pretrained,
                                                   checkpoint_path, scriptable,
                                                   exportable, no_jit,
                                                   **kwargs)

            # need fix: delete this except after pytorch 1.7 update in all production
            # (dlc, dsw, studio, ev_predict_py3)
            except Exception:
                print(
                    f'Error: Fail to create {model_name} with (pretrained={pretrained}, checkpoint_path={checkpoint_path} ...)'
                )
                print(
                    f'Try to create {model_name} with pretrained=False, checkpoint_path=None and default params'
                )
                self.model = timm.create_model(model_name, False, '', None,
                                               None, None, **kwargs)

        # facebook model wrapper
        if model_name in _MODEL_MAP:
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

                    # for some model strict = False still failed when model doesn't exactly match
                    try:
                        self.model.load_state_dict(state_dict, strict=False)
                    except Exception:
                        print('load for model_name not all right')
                else:
                    print('%s not in evtorch modelzoo!' % model_name)

    def init_weights(self, pretrained=None):
        # pretrained is the path of pretrained model offered by easycv
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(
                self.model,
                pretrained,
                map_location=torch.device('cpu'),
                strict=False,
                logger=logger)
        else:
            # init by timm
            pass

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
