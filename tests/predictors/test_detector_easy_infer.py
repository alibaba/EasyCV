# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import os, io
import unittest
import tempfile
import numpy as np
from PIL import Image
import torch
import torchvision
import cv2

from tests.ut_config import (PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT,
                             DET_DATA_SMALL_COCO_LOCAL)


class YoloXEasyInferTest(unittest.TestCase):
    img = os.path.join(DET_DATA_SMALL_COCO_LOCAL, 'val2017/000000522713.jpg')

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_easy_infer(self):
        # load img
        img = cv2.imread(self.img)
        img = torch.tensor(img).unsqueeze(0).cuda()

        # load model
        model_path = PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT
        preprocess_path = '.'.join(model_path.split('.')[:-1] + ['preprocess'])
        with io.open(preprocess_path, 'rb') as infile:
            preprocess = torch.jit.load(infile)
        with io.open(model_path, 'rb') as infile:
            model = torch.jit.load(infile)

        # preporcess with the exported model or use your own preprocess func
        img, img_info = preprocess(img)

        # forward with nms [b,c,h,w] -> List[[n,7]]
        # n means the predicted box num of each img
        # 7 means [x1,y1,x2,y2,obj_conf,cls_conf,cls]
        outputs = model(img)

        # postprocess the output information into dict or your own data structure
        # slice box,score,class & rescale box
        detection_boxes = []
        detection_scores = []
        detection_classes = []
        bboxes = outputs[0][:, 0:4]
        bboxes /= img_info['scale_factor'][0]
        detection_boxes.append(bboxes.cpu().detach().numpy())
        detection_scores.append(
            (outputs[0][:, 4] * outputs[0][:, 5]).cpu().detach().numpy())
        detection_classes.append(
            outputs[0][:, 6].cpu().detach().numpy().astype(np.int32))

        final_outputs = {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
        }

        assert (len(final_outputs['detection_boxes']) == len(
            final_outputs['detection_scores']))
        assert (len(final_outputs['detection_scores']) == len(
            final_outputs['detection_classes']))


if __name__ == '__main__':
    unittest.main()
