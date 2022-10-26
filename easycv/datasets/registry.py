# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.utils.registry import Registry
from .utils.download_util import DownLoadDataFile

DATASOURCES = Registry('datasource')
DATASETS = Registry('dataset')
DALIDATASETS = Registry('dalidataset')
PIPELINES = Registry('pipeline')
SAMPLERS = Registry('sampler')
DOWNLOAD = DownLoadDataFile()