# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging

from easycv.models import build_model
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--backbone_key',
        type=str,
        default='backbone',
        help='backbone key name of model')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv_config_fromfile(args.config)

    # dynamic adapt mmdet models
    dynamic_adapt_for_mmlab(cfg)

    model = build_model(cfg.model)

    unit = 1e6

    num_params = sum(p.numel() for p in model.parameters()) / unit
    num_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad) / unit

    if hasattr(model, args.backbone_key):
        num_backbone_params = sum(
            p.numel()
            for p in getattr(model, args.backbone_key).parameters()) / unit
        num_backbone_grad_params = sum(
            p.numel() for p in getattr(model, args.backbone_key).parameters()
            if p.requires_grad) / unit
        print('Number of backbone parameters: {:.5g} M'.format(
            num_backbone_params))
        print('Number of backbone parameters requiring grad: {:.5g} M'.format(
            num_backbone_grad_params))
    else:
        logging.warning(
            f'The backbone with the key ``{args.backbone_key}`` was not found in the model !'
        )

    print('Number of total parameters: {:.5g} M'.format(num_params))
    print('Number of total parameters requiring grad: {:.5g} M'.format(
        num_grad_params))


if __name__ == '__main__':
    main()
