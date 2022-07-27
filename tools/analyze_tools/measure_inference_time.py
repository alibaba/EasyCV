# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse

import numpy as np
import torch
import tqdm
from mmcv.parallel import scatter_kwargs
from torch.backends import cudnn

from easycv.datasets.builder import build_dataset
from easycv.datasets.loader import build_dataloader
from easycv.models.builder import build_model
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab


def parse_args():
    parser = argparse.ArgumentParser(
        description='EasyCV model memory and inference_time test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--repeat_num', default=300, type=int, help='repeat number')
    parser.add_argument(
        '--warmup_num', default=100, type=int, help='warm up number')
    parser.add_argument(
        '--gpu',
        default='0',
        type=str,
        choices=['0', '1', '2', '3', '4', '5', '6', '7'])

    args = parser.parse_args()
    return args


def main():
    cudnn.benchmark = True

    args = parse_args()

    cfg = mmcv_config_fromfile(args.config)

    # dynamic adapt mmdet models
    dynamic_adapt_for_mmlab(cfg)

    device = torch.device('cuda:{}'.format(args.gpu))
    model = build_model(cfg.model).to(device)
    model.eval()

    cfg.data.val.pop('imgs_per_gpu', None)  # pop useless params
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=0,
    )

    # Set up cuda events for measuring time. This is PyTorch's official recommended interface and should theoretically be the most reliable.
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    # Initialize a time container.
    timings = np.zeros((args.repeat_num, 1))

    with torch.no_grad():
        for idx, data in zip(tqdm.trange(args.repeat_num), data_loader):
            _, kwargs = scatter_kwargs(None, data, [int(args.gpu)])
            inputs = kwargs[0]
            inputs.update(dict(mode='test'))
            # GPU may be hibernated to save energy at ordinary times, so it needs to be preheated.
            if idx < args.warmup_num:
                if idx == 0:
                    print('Start warm up ...')
                _ = model(**inputs)
                continue

            if idx == args.warmup_num:
                print('Warm up end, start to record time...')

            starter.record()
            _ = model(**inputs)
            ender.record()
            torch.cuda.synchronize()  # Wait for the GPU task to complete.
            curr_time = starter.elapsed_time(
                ender)  # The time between starter and ender, in milliseconds.
            timings[idx] = curr_time

    avg = timings.sum() / args.repeat_num
    print('Cuda memory: {}'.format(torch.cuda.memory_summary(device)))
    print('\ninference average time={}ms\n'.format(avg))


if __name__ == '__main__':
    main()
