# Copyright (c) Alibaba, Inc. and its affiliates.
# flake8: noqa
# isort:skip_file
import os
os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'

from .version import __version__, short_version

__all__ = ['__version__', 'short_version']
