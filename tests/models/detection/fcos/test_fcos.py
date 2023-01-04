# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from easycv.predictors.detector import DetectionPredictor


class FCOSTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_fcos(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/fcos_epoch_12.pth'
        config_path = 'configs/detection/fcos/fcos_r50_torch_1x_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        model = DetectionPredictor(model_path, config_path)
        output = model(img)[0]
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            tmp_save_path = tmp_file.name
            model.visualize(img, output, out_file=tmp_save_path)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes']), 7)
        self.assertEqual(len(output['detection_scores']), 7)
        self.assertEqual(len(output['detection_classes']), 7)

        assert_array_almost_equal(
            output['detection_classes'].tolist(),
            np.array([2, 2, 2, 2, 2, 2, 2], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([
                0.7142099, 0.61647004, 0.5857586, 0.5839255, 0.5378273,
                0.5127002, 0.5077106
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
            np.array([[294.96497, 116.47906, 378.7294, 149.90738],
                      [480.34415, 110.31671, 523.0271, 130.33409],
                      [398.22247, 110.64816, 433.01566, 133.1527],
                      [608.2505, 111.9937, 636.7885, 137.0966],
                      [591.46234, 109.84667, 619.6144, 126.97513],
                      [431.47202, 104.88086, 482.88544, 131.95964],
                      [189.96198, 108.948654, 297.10025, 154.80592]]),
            decimal=1)

    @unittest.skip('skip bytetrack unittest')
    def test_bytetrack(self):
        from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/fcos_epoch_12.pth'
        config_path = 'configs/detection/fcos/fcos_r50_torch_1x_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        model = DetectionPredictor(model_path, config_path)
        output = model(img)[0]
        tracker = BYTETracker(
            det_high_thresh=0.2,
            det_low_thresh=0.05,
            match_thresh=1.0,
            match_thresh_second=1.0,
            match_thresh_init=1.0,
            track_buffer=2,
            frame_rate=25)
        track_result = tracker.update(output['detection_boxes'],
                                      output['detection_scores'],
                                      output['detection_classes'])

        assert_array_almost_equal(
            track_result['track_bboxes'],
            np.array([[
                1.00000000e+00, 2.94000000e+02, 1.16000000e+02, 3.78000000e+02,
                1.49000000e+02, 7.14209914e-01
            ],
                      [
                          2.00000000e+00, 4.80000000e+02, 1.10000000e+02,
                          5.23000000e+02, 1.30000000e+02, 6.16470039e-01
                      ],
                      [
                          3.00000000e+00, 3.98000000e+02, 1.10000000e+02,
                          4.33000000e+02, 1.33000000e+02, 5.85758626e-01
                      ],
                      [
                          4.00000000e+00, 6.08000000e+02, 1.11000000e+02,
                          6.36000000e+02, 1.37000000e+02, 5.83925486e-01
                      ],
                      [
                          5.00000000e+00, 5.91000000e+02, 1.09000000e+02,
                          6.19000000e+02, 1.26000000e+02, 5.37827313e-01
                      ],
                      [
                          6.00000000e+00, 4.31000000e+02, 1.04000000e+02,
                          4.82000000e+02, 1.31000000e+02, 5.12700200e-01
                      ],
                      [
                          7.00000000e+00, 1.89000000e+02, 1.08000000e+02,
                          2.97000000e+02, 1.54000000e+02, 5.07710576e-01
                      ]]),
            decimal=1)


if __name__ == '__main__':
    unittest.main()
