from distutils.version import LooseVersion

import torch
from torch.optim import *

from .adai import Adai
from .adan import Adan
from .builder import build_optimizer_constructor
from .lamb import Lamb
from .lars import LARS
from .layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor
from .ranger import Ranger

if LooseVersion(torch.__version__) <= LooseVersion('1.9.0'):
    from .adam import AdamW
