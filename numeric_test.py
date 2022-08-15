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

if __name__=='__main__':
    if 1:
        img_path = '/apsara/xinyi.zxy/data/coco/val2017/000000254016.jpg'
        from easycv.predictors import TorchYoloXPredictor
        img = Image.open(img_path)

        pred = TorchYoloXPredictor('models/output.pt')
        m = pred.predict([img])
        print("fucking m :", m)

        pred0 = TorchYoloXPredictor('models/output_bs1.blade.jit')
        for i in range(10):
            m0 = pred0.predict([img])
        with timeit_context('m0 speed test'):
            for i in range(100):
                m0 = pred0.predict([img]) 
        print("fucking m0:", m0)
        
        pred1 = TorchYoloXPredictor('models/output_bs1_e2e.blade.jit')
        for i in range(10):
            m1 = pred1.predict([img])
        with timeit_context('m1 speed test'):
            for i in range(100):
                m1 = pred1.predict([img])       
        print("fucking m1:", m1)

        # pred2 = TorchYoloXPredictor('models/output_bs1_e2e.blade.jit')
        # m2 = pred2.predict([img])
        # print("fucking m2:", m2)

        # pred3 = TorchYoloXPredictor('models/output_bs1_e2e_f005.blade.jit')
        # m3 = pred3.predict([img])
        # print("fucking m3:", m3)

        # pred4 = TorchYoloXPredictor('models/output_trtnms.pt')
        # m4 = pred4.predict([img]) 
        # print("fucking m4:", m4)

        pred5 = TorchYoloXPredictor(model_path='models/output_bs1_noe2e_f005_trtnms.blade.blade', use_trt_nms=True)
        # m5 = pred5.predict([img]) 
        for i in range(10):
            m5 = pred5.predict([img])
        with timeit_context('m5 speed test'):
            for i in range(100):
                m5 = pred5.predict([img])     
        print("fucking m5:", m5)

        pred6 = TorchYoloXPredictor(model_path='models/output_bs1_e2e_f005_trtnms.blade.blade', use_trt_nms=True)
        # m5 = pred5.predict([img]) 
        for i in range(10):
            m6 = pred6.predict([img])
        with timeit_context('m6 speed test'):
            for i in range(100):
                m6 = pred5.predict([img])     
        print("fucking m6:", m6)