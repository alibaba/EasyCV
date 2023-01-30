# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import os
import unittest
import tempfile
import numpy as np
from PIL import Image
from easycv.predictors.detector import DetectionPredictor, YoloXPredictor, TorchYoloXPredictor
from tests.ut_config import (PRETRAINED_MODEL_YOLOXS_EXPORT,
                             PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_JIT,
                             PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT,
                             PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT_B2,
                             DET_DATA_SMALL_COCO_LOCAL)
from numpy.testing import assert_array_almost_equal


class YoloXPredictorTest(unittest.TestCase):
    img = os.path.join(DET_DATA_SMALL_COCO_LOCAL, 'val2017/000000522713.jpg')

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _assert_results(self, results):
        self.assertEqual(results['ori_img_shape'], [480, 640])
        self.assertListEqual(results['detection_classes'].tolist(),
                             np.array([13, 8, 8, 8], dtype=np.int32).tolist())
        self.assertListEqual(results['detection_class_names'],
                             ['bench', 'boat', 'boat', 'boat'])
        assert_array_almost_equal(
            results['detection_scores'],
            np.array([0.92335737, 0.59416807, 0.5567955, 0.55368793],
                     dtype=np.float32),
            decimal=2)
        assert_array_almost_equal(
            results['detection_boxes'],
            np.array([[408.1708, 285.11456, 561.84924, 356.42285],
                      [438.88098, 264.46606, 467.07275, 271.76355],
                      [510.19467, 268.46664, 528.26935, 273.37192],
                      [480.9472, 269.74115, 502.00842, 274.85553]]),
            decimal=1)

    def _base_test_single(self, model_path, inputs):
        predictor = YoloXPredictor(model_path=model_path, score_thresh=0.5)
        outputs = predictor(inputs)
        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        self._assert_results(output)

    def _base_test_batch(self,
                         model_path,
                         inputs,
                         num_samples,
                         batch_size,
                         num_parallel=8):
        assert isinstance(inputs, list) and len(inputs) == 1

        predictor = YoloXPredictor(
            model_path=model_path,
            score_thresh=0.5,
            batch_size=batch_size,
            num_parallel=num_parallel)
        outputs = predictor(inputs * num_samples)

        self.assertEqual(len(outputs), num_samples)
        for output in outputs:
            self._assert_results(output)

    def test_single_raw(self):
        model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        inputs = [np.asarray(Image.open(self.img))]
        self._base_test_single(model_path, inputs)

    def test_batch_raw(self):
        model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        inputs = [np.asarray(Image.open(self.img))]
        self._base_test_batch(model_path, inputs, num_samples=3, batch_size=2)

    def test_single_jit_nopre_notrt(self):
        jit_path = PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_JIT
        self._base_test_single(jit_path, self.img)

    def test_batch_jit_nopre_notrt(self):
        jit_path = PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_JIT
        self._base_test_batch(
            jit_path, [self.img], num_samples=2, batch_size=1)

    def test_single_jit_pre_trt(self):
        jit_path = PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT
        self._base_test_single(jit_path, [self.img])

    def test_batch_jit_pre_trt(self):
        jit_path = PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT_B2
        self._base_test_batch(
            jit_path, [self.img], num_samples=4, batch_size=2, num_parallel=1)

    def test_single_raw_TorchYoloXPredictor(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        input_data_list = [np.asarray(Image.open(self.img))]
        predictor = TorchYoloXPredictor(
            model_path=detection_model_path, score_thresh=0.5)
        output = predictor(input_data_list)[0]
        self._assert_results(output)


class DetectionPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _detection_detector_assert(self, output):
        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('detection_masks', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes']), 33)
        self.assertEqual(len(output['detection_scores']), 33)
        self.assertEqual(len(output['detection_classes']), 33)

        self.assertListEqual(
            output['detection_classes'].tolist(),
            np.array([
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 7, 7, 13, 13, 13, 56
            ],
                     dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([
                0.9975854158401489, 0.9965696334838867, 0.9922919869422913,
                0.9833580851554871, 0.983080267906189, 0.970454752445221,
                0.9701289534568787, 0.9649872183799744, 0.9642795324325562,
                0.9642238020896912, 0.9529680609703064, 0.9403366446495056,
                0.9391788244247437, 0.8941807150840759, 0.8178097009658813,
                0.8013413548469543, 0.6677654385566711, 0.3952914774417877,
                0.33463895320892334, 0.32501447200775146, 0.27323535084724426,
                0.20197080075740814, 0.15607696771621704, 0.1068163588643074,
                0.10183875262737274, 0.09735643863677979, 0.06559795141220093,
                0.08890066295862198, 0.076363705098629, 0.9954648613929749,
                0.9212945699691772, 0.5224372148513794, 0.20555885136127472
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
            np.array([[
                294.22674560546875, 116.6078109741211, 379.4328918457031,
                150.14097595214844
            ],
                      [
                          482.6017761230469, 110.75955963134766,
                          522.8798828125, 129.71286010742188
                      ],
                      [
                          167.06460571289062, 109.95974731445312,
                          212.83975219726562, 140.16102600097656
                      ],
                      [
                          609.2930908203125, 113.13909149169922,
                          637.3115844726562, 136.4690704345703
                      ],
                      [
                          191.185791015625, 111.1408920288086, 301.31689453125,
                          155.7731170654297
                      ],
                      [
                          431.2244873046875, 106.19962310791016,
                          483.860595703125, 132.21627807617188
                      ],
                      [
                          267.48358154296875, 105.5920639038086,
                          325.2832336425781, 127.11176300048828
                      ],
                      [
                          591.2138671875, 110.29329681396484,
                          619.8524169921875, 126.1990966796875
                      ],
                      [
                          0.0, 110.7026596069336, 61.487945556640625,
                          146.33018493652344
                      ],
                      [
                          555.9155883789062, 110.03486633300781,
                          591.7050170898438, 127.06097412109375
                      ],
                      [
                          60.24559783935547, 94.12760162353516,
                          85.63741302490234, 106.66705322265625
                      ],
                      [
                          99.02665710449219, 90.53657531738281,
                          118.83953094482422, 101.18717956542969
                      ],
                      [
                          396.30438232421875, 111.59194946289062,
                          431.559814453125, 133.96914672851562
                      ],
                      [
                          83.81543731689453, 89.65665435791016,
                          99.9166259765625, 98.25627899169922
                      ],
                      [
                          139.29647827148438, 96.68000793457031,
                          165.22410583496094, 105.60000610351562
                      ],
                      [
                          67.27152252197266, 89.42798614501953,
                          83.25617980957031, 98.0460205078125
                      ],
                      [
                          223.74176025390625, 98.68321990966797,
                          250.42506408691406, 109.32588958740234
                      ],
                      [
                          136.7582244873047, 96.51412963867188,
                          152.51190185546875, 104.73160552978516
                      ],
                      [
                          221.71812438964844, 97.86445617675781,
                          238.9705810546875, 106.96803283691406
                      ],
                      [
                          135.06964111328125, 91.80916595458984, 155.24609375,
                          102.20686340332031
                      ],
                      [
                          169.11180114746094, 97.53628540039062,
                          182.88504028320312, 105.95404815673828
                      ],
                      [
                          133.8811798095703, 91.00375366210938,
                          145.35507202148438, 102.3780288696289
                      ],
                      [
                          614.2507934570312, 102.19828796386719,
                          636.5692749023438, 112.59198760986328
                      ],
                      [
                          35.94759750366211, 91.7213363647461,
                          70.38274383544922, 117.19855499267578
                      ],
                      [
                          554.6401977539062, 115.18976593017578,
                          562.0255737304688, 127.4429931640625
                      ],
                      [
                          39.07550811767578, 92.73261260986328,
                          85.36636352539062, 106.73953247070312
                      ],
                      [
                          200.85513305664062, 93.00469970703125,
                          219.73086547851562, 107.99642181396484
                      ],
                      [
                          0.0, 111.18904876708984, 61.7393684387207,
                          146.72547912597656
                      ],
                      [
                          191.88568115234375, 111.09577178955078,
                          299.4097900390625, 155.14639282226562
                      ],
                      [
                          221.06834411621094, 176.6427001953125,
                          458.3475341796875, 378.89300537109375
                      ],
                      [
                          372.7131652832031, 135.51429748535156,
                          433.2494201660156, 188.0106658935547
                      ],
                      [
                          52.19819641113281, 110.3646011352539,
                          70.95110321044922, 120.10567474365234
                      ],
                      [
                          376.1671447753906, 133.6930694580078,
                          432.2721862792969, 187.99481201171875
                      ]]),
            decimal=1)

    def test_detection_detector_single(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/vitdet/vit_base/epoch_100_export.pth'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        vitdet = DetectionPredictor(model_path, score_threshold=0.0)
        output = vitdet(img)
        output = output[0]
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            tmp_save_path = tmp_file.name
            vitdet.visualize(img, output, out_file=tmp_save_path)
        self._detection_detector_assert(output)

    def test_detection_detector_batch(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/vitdet/vit_base/epoch_100_export.pth'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        vitdet = DetectionPredictor(
            model_path, score_threshold=0.0, batch_size=2)
        num_samples = 3
        images = [img] * num_samples
        outputs = vitdet(images)
        self.assertEqual(len(outputs), num_samples)
        for output in outputs:
            with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
                tmp_save_path = tmp_file.name
                vitdet.visualize(img, output, out_file=tmp_save_path)
            self._detection_detector_assert(output)


if __name__ == '__main__':
    unittest.main()
