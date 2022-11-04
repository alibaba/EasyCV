# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from easycv.predictors.builder import build_predictor
from easycv.utils.config_tools import mmcv_config_fromfile


class HydraAttentionTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_hydraAttention(self):
        checkpoint = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hydra_attention/deit_base_patch16_224%20(Hydra%20Attention%20%5B12%20layers%5D).pth'
        config_file = 'configs/classification/imagenet/deit/imagenet_deit_base_hydra_layer12_patch16_224_jpg.py'
        img_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/imagenet_demo.JPEG'
        cfg = mmcv_config_fromfile(config_file)
        predict_op = build_predictor(
            dict(
                **cfg.predict,
                model_path=checkpoint,
                config_file=config_file,
                label_map_path=None,
                pil_input=False))

        results = predict_op([img_path])[0]

        self.assertIn('class', results)
        self.assertEqual(len(results['class']), 1)
        self.assertEqual(int(results['class'][0]), 234)


if __name__ == '__main__':
    unittest.main()
