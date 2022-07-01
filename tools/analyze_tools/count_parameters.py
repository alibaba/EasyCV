# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse

from easycv.models import build_model
from easycv.utils.config_tools import mmcv_config_fromfile


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv_config_fromfile(args.config)

    model = build_model(cfg.model)

    unit = 1e6

    num_params = sum(p.numel() for p in model.parameters()) / unit
    num_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad) / unit
    num_backbone_params = sum(p.numel()
                              for p in model.backbone.parameters()) / unit
    num_backbone_grad_params = sum(p.numel()
                                   for p in model.backbone.parameters()
                                   if p.requires_grad) / unit
    print(
        'Number of backbone parameters: {:.5g} M'.format(num_backbone_params))
    print('Number of backbone parameters requiring grad: {:.5g} M'.format(
        num_backbone_grad_params))
    print('Number of total parameters: {:.5g} M'.format(num_params))
    print('Number of total parameters requiring grad: {:.5g} M'.format(
        num_grad_params))


if __name__ == '__main__':
    main()
