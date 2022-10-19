# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from distutils.version import LooseVersion

import numpy as np


def check_numpy():

    def require(version):
        if LooseVersion(np.__version__) < LooseVersion(version):
            raise ImportError(
                f'numpy version should be greater than {version}')

    if sys.version_info.minor == 6:
        require('1.19.5')
    else:
        require('1.20.0')
