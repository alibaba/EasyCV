# Copyright (c) Alibaba, Inc. and its affiliates.


def parse_pipleline(test_pipeline):
    # default
    target_size = (640, 640)
    keep_ratio = True
    pad_val = 114
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    to_rgb = True

    for i in range(len(test_pipeline)):
        if test_pipeline[i]['type'] == 'MMResize':
            target_size = test_pipeline[i]['img_scale']
            keep_ratio = test_pipeline[i]['keep_ratio']

        if test_pipeline[i]['type'] == 'MMPad':
            pad_val = int(test_pipeline[i]['pad_val'][0])

        if test_pipeline[i]['type'] == 'MMNormalize':
            mean = test_pipeline[i]['mean']
            std = test_pipeline[i]['std']
            to_rgb = test_pipeline[i]['to_rgb']

    return target_size, keep_ratio, pad_val, mean, std, to_rgb
