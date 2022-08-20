import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose

from easycv.datasets.registry import PIPELINES
from easycv.models import build_model
from easycv.models.detection.detectors.yolox.postprocess import \
    create_tensorrt_postprocess
from easycv.models.detection.utils import postprocess
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.registry import build_from_cfg


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
        for i in range(1000):
            m0 = pred.predict([img])
    print(m0[0]['detection_classes'])
    print(m0[0]['detection_scores'])
    print(m0[0]['detection_boxes'])
    print(len(m0[0]['detection_classes']))


if __name__ == '__main__':
    if 1:
        img_path = '/apsara/xinyi.zxy/data/coco/val2017/000000037777.jpg'
        from easycv.predictors import TorchYoloXPredictor

        img = Image.open(img_path)

        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300.pt', img)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300_nopre_notrt.pt.jit', img, False)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300_nopre_trt.pt.jit', img, True)  # jit ??
        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300_pre_notrt.pt.jit', img, False)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300_pre_trt.pt.jit', img, True)
        #
        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300_nopre_notrt.pt.blade', img, False)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300_nopre_trt.pt.blade', img, True)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300_pre_notrt.pt.blade', img, False)
        # model_speed_test('/apsara/xinyi.zxy/pretrain/infer_yolox/epoch_300_pre_trt.pt.blade', img, True)

        model_speed_test('/apsara/xinyi.zxy/pretrain/base_export/s.pt.blade', img, False)
        model_speed_test('/apsara/xinyi.zxy/pretrain/base_export/m.pt.blade', img, False)
        model_speed_test('/apsara/xinyi.zxy/pretrain/base_export/l.pt.blade', img, False)
        model_speed_test('/apsara/xinyi.zxy/pretrain/base_export/x.pt.blade', img, False)
