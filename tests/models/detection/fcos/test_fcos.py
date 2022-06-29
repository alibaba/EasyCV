# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch
from mmcv.parallel import collate, scatter
from numpy.testing import assert_array_almost_equal
from torchvision.transforms import Compose

from easycv.datasets.registry import PIPELINES
from easycv.datasets.utils import replace_ImageToTensor
from easycv.models import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab
from easycv.utils.registry import build_from_cfg


class FCOSTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def init_fcos(self, model_path, config_path):
        self.model_path = model_path

        self.cfg = mmcv_config_fromfile(config_path)

        # dynamic adapt mmdet models
        dynamic_adapt_for_mmlab(self.cfg)

        # modify model_config
        if self.cfg.model.head.test_cfg.get('max_per_img', None):
            self.cfg.model.head.test_cfg.max_per_img = 10

        # build model
        self.model = build_model(self.cfg.model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        map_location = 'cpu' if self.device == 'cpu' else 'cuda'
        self.ckpt = load_checkpoint(
            self.model, self.model_path, map_location=map_location)

        self.model.to(self.device)
        self.model.eval()

        self.CLASSES = self.cfg.CLASSES

    def predict(self, imgs):
        """Inference image(s) with the detector.
        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        Returns:
            If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
        """

        if isinstance(imgs, (list, tuple)):
            is_batch = True
        else:
            imgs = [imgs]
            is_batch = False

        cfg = self.cfg
        device = next(self.model.parameters()).device  # model device

        if isinstance(imgs[0], np.ndarray):
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.val.pipeline.insert(
                0,
                dict(
                    type='LoadImageFromWebcam',
                    file_client_args=dict(backend='http')))
        else:
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.val.pipeline.insert(
                0,
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='http')))

        cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)

        transforms = []
        for transform in cfg.data.val.pipeline:
            if 'img_scale' in transform:
                transform['img_scale'] = tuple(transform['img_scale'])
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                transforms.append(transform)
            elif callable(transform):
                transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')
        test_pipeline = Compose(transforms)

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]

        # forward the model
        with torch.no_grad():
            results = self.model(mode='test', **data)

        return results

    def test_fcos(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/epoch_12.pth'
        config_path = 'configs/detection/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        self.init_fcos(model_path, config_path)
        output = self.predict(img)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 10)
        self.assertEqual(len(output['detection_scores'][0]), 10)
        self.assertEqual(len(output['detection_classes'][0]), 10)

        print(output['detection_boxes'][0].tolist())
        print(output['detection_scores'][0].tolist())
        print(output['detection_classes'][0].tolist())

        self.assertListEqual(
            output['detection_classes'][0].tolist(),
            np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 13], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0],
            np.array([
                0.6641181707382202, 0.6135501265525818, 0.5985610485076904,
                0.5694775581359863, 0.5586040616035461, 0.5209507942199707,
                0.5056729912757874, 0.4943872094154358, 0.4850597083568573,
                0.45443734526634216
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0],
            np.array([[
                295.5196228027344, 116.56035614013672, 380.0883483886719,
                150.24908447265625
            ],
                      [
                          190.57131958007812, 108.96343231201172,
                          297.7738037109375, 154.69515991210938
                      ],
                      [
                          480.5726013183594, 110.4341812133789,
                          522.8551635742188, 129.9452667236328
                      ],
                      [
                          431.1232604980469, 105.17676544189453,
                          483.89617919921875, 131.85870361328125
                      ],
                      [
                          398.6544494628906, 110.90837860107422,
                          432.6370849609375, 132.89173889160156
                      ],
                      [
                          609.3126831054688, 111.62432861328125,
                          635.4577026367188, 137.03529357910156
                      ],
                      [
                          98.66332244873047, 89.88417053222656,
                          118.9398422241211, 101.25397491455078
                      ],
                      [
                          167.9045867919922, 109.57560729980469,
                          209.74375915527344, 139.98898315429688
                      ],
                      [
                          591.0496826171875, 110.55867767333984,
                          619.4395751953125, 126.65755462646484
                      ],
                      [
                          218.92051696777344, 177.0509033203125,
                          455.8321838378906, 385.0356140136719
                      ]]),
            decimal=1)


if __name__ == '__main__':
    unittest.main()
