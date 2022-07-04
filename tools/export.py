# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
from __future__ import division
import argparse
import os
import os.path as osp
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(
    os.path.abspath(
        osp.join(os.path.dirname(os.path.dirname(__file__)), '../')))

import time

import requests

from easycv import __version__
from easycv.apis.export import export
from easycv.file import io
from easycv.utils.logger import get_root_logger
# from mmcv import Config
from easycv.utils.config_tools import (CONFIG_TEMPLATE_ZOO,
                                       mmcv_config_fromfile, rebuild_config)
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab


def parse_args():
    parser = argparse.ArgumentParser(description='export a model')
    parser.add_argument(
        'config', help='config file path', type=str, default=None)
    parser.add_argument(
        'ckpt_path', type=str, help='checkpoint to be exported')
    parser.add_argument(
        'export_path', type=str, help='file to store the exported model')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        help=
        'parameterize param when user specific choose a model config template like CLASSIFICATION: classification.py'
    )
    parser.add_argument(
        '--user_config_params',
        nargs=argparse.REMAINDER,
        default=None,
        help='modify config options using the command-line')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.model_type is not None:
        assert args.model_type in CONFIG_TEMPLATE_ZOO, 'model_type must be in [%s]' % (
            ', '.join(CONFIG_TEMPLATE_ZOO.keys()))
        print('model_type=%s, config file will be replaced by %s' %
              (args.model_type, CONFIG_TEMPLATE_ZOO[args.model_type]))
        args.config = CONFIG_TEMPLATE_ZOO[args.model_type]

    print(args.config)

    if args.config.startswith('http'):

        r = requests.get(args.config)
        # download config in current dir
        tpath = args.config.split('/')[-1]
        while not osp.exists(tpath):
            try:
                with open(tpath, 'wb') as code:
                    code.write(r.content)
            except:
                pass

        args.config = tpath

    cfg = mmcv_config_fromfile(args.config)

    if args.user_config_params is not None:
        assert args.model_type is not None, 'model_type must be setted'
        # rebuild config by user config params
        cfg = rebuild_config(cfg, args.user_config_params)

    # dynamic adapt mmdet models
    dynamic_adapt_for_mmlab(cfg)

    # check oss_config and init oss io
    if cfg.get('oss_io_config', None) is not None:
        io.access_oss(**cfg.oss_io_config)

    # init the logger before other steps
    logger = get_root_logger(log_level=cfg.log_level)

    export(cfg, args.ckpt_path, args.export_path)


if __name__ == '__main__':
    main()
