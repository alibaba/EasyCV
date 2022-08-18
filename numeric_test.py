from easycv.models.detection.detectors.yolox.postprocess import create_tensorrt_postprocess
import torch
from torchvision.transforms import Compose

from easycv.models import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.registry import build_from_cfg
from easycv.datasets.registry import PIPELINES
from easycv.models.detection.utils import postprocess

import sys
import numpy as np
from PIL import Image
import time

from contextlib import contextmanager


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


def model_speed_test(name, img, use_trt_efficientnms=False):
    pred = TorchYoloXPredictor(name, use_trt_efficientnms=use_trt_efficientnms)
    for i in range(10):
        m0 = pred.predict([img])
    with timeit_context('{} speed test'.format(name)):
        for i in range(100):
            m0 = pred.predict([img])
    print(m0[0]['detection_classes'])
    print(m0[0]['detection_scores'])


if __name__ == '__main__':
    if 1:
        img_path = '/apsara/xinyi.zxy/data/coco/val2017/000000254016.jpg'
        from easycv.predictors import TorchYoloXPredictor

        img = Image.open(img_path)

        # model_speed_test('/apsara/xinyi.zxy/pretrain/yolox_s/epoch_300.pt', img)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/yolox_s/epoch_300_j_noe2e.pt.jit', img, False)
        # # model_speed_test('/apsara/xinyi.zxy/pretrain/yolox_s/epoch_300_j_noe2e_trt.pt.jit', img, True)  # jit ??
        # model_speed_test('/apsara/xinyi.zxy/pretrain/yolox_s/epoch_300_j_e2e_notrt.pt.jit', img, False)
        # #model_speed_test('/apsara/xinyi.zxy/pretrain/yolox_s/epoch_300_j_e2e_trt.pt.jit', img, True)

        model_speed_test('/apsara/xinyi.zxy/pretrain/yolox_s/epoch_300_b_noe2e.pt.blade', img, False)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/yolox_s/epoch_300_b_noe2e_trt.pt.blade', img, True)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/yolox_s/epoch_300_b_e2e_trt.pt.blade', img, True)

