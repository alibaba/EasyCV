# Copyright (c) Alibaba, Inc. and its affiliates.
from . import data_sources  # pylint: disable=unused-import
from . import pipelines  # pylint: disable=unused-import
from .top_down import PoseTopDownDataset

__all__ = ['PoseTopDownDataset']
