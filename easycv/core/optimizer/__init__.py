from distutils.version import LooseVersion

import torch
from torch.optim import *

from .lars import LARS
from .ranger import Ranger

if LooseVersion(torch.__version__) <= LooseVersion('1.9.0'):
    from .adam import AdamW
