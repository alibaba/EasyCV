# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

from easycv.models import build_model
from easycv.utils.config_tools import mmcv_config_fromfile

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='EasyCV model memory and inference_time test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'gpu', type=str, choices=['0', '1', '2', '3', '4', '5', '6', '7'])

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv_config_fromfile(args.config)

    device = torch.device('cuda:{}'.format(args.gpu))
    model = build_model(cfg.model).to(device)
    repetitions = 300

    dummy_input = torch.rand(1, 3, 224, 224).to(device)

    # Preheat: GPU may be hibernated to save energy at ordinary times, so it needs to be preheated.
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model.forward_test(dummy_input)

    # Synchronize Waits for all GPU tasks to complete before returning to the CPU main thread.
    torch.cuda.synchronize()

    # Set up cuda events for measuring time. This is PyTorch's official recommended interface and should theoretically be the most reliable.
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True)
    # Initialize a time container.
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model.forward_test(dummy_input)
            ender.record()
            torch.cuda.synchronize()  # Wait for the GPU task to complete.
            curr_time = starter.elapsed_time(
                ender)  # The time between starter and ender, in milliseconds.
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print(torch.cuda.memory_summary(device))
    print('\navg={}\n'.format(avg))


if __name__ == '__main__':
    main()
