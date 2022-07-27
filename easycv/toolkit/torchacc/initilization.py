# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from distutils.version import LooseVersion

import torch
import torchacc.torch_xla.core.xla_model as xm

from .convert_ops import convert_timm_ops, convert_torch_ops_to_torchacc


def patch_ops():
    convert_timm_ops()
    convert_torch_ops_to_torchacc()


def torchacc_init():
    assert LooseVersion(torch.__version__) >= LooseVersion(
        '1.10.0'), 'torchacc only supports torch versions greater than 1.10'

    if xm.xrt_world_size() == 1:
        os.environ['GPU_NUM_DEVICES'] = '1'

    device = xm.xla_device()
    xm.set_replication(device, [device])

    patch_ops()
