# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from easycv.predictors.detector import DetectionPredictor


class DETRTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_detr(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/detr/epoch_150.pth'
        config_path = 'configs/detection/detr/detr_r50_8x2_150e_coco.py'
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
        self.assertEqual(len(output['detection_boxes']), 26)
        self.assertEqual(len(output['detection_scores']), 26)
        self.assertEqual(len(output['detection_classes']), 26)

        assert_array_almost_equal(
            output['detection_classes'].tolist(),
            np.array([
                2, 2, 2, 2, 2, 13, 2, 2, 2, 7, 56, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2
            ],
                     dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([
                0.58311516, 0.98532575, 0.50060254, 0.9802161, 0.95413357,
                0.98143035, 0.989082, 0.94934535, 0.652008, 0.5401012,
                0.5485139, 0.5970404, 0.6823337, 0.98559755, 0.5903073,
                0.98136836, 0.98148626, 0.50042206, 0.58529335, 0.8264537,
                0.9733429, 0.7118396, 0.95125425, 0.9736388, 0.9338273,
                0.98050916
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
            np.array([[
                1.32131638e+02, 9.08366165e+01, 1.51008240e+02, 1.01831055e+02
            ], [
                1.89690186e+02, 1.08048561e+02, 2.96801422e+02, 1.54441940e+02
            ], [
                1.63976013e+02, 9.29504929e+01, 1.80869934e+02, 1.02616295e+02
            ], [
                1.65771057e+02, 1.08236237e+02, 2.08613281e+02, 1.36434570e+02
            ], [
                5.64804199e+02, 1.08129990e+02, 5.93914856e+02, 1.26268921e+02
            ], [
                2.18924438e+02, 1.77140930e+02, 4.59107849e+02, 3.81113098e+02
            ], [
                3.97366943e+02, 1.10411560e+02, 4.36520844e+02, 1.33168503e+02
            ], [
                5.76233597e+01, 9.05034256e+01, 8.22042923e+01, 1.03573486e+02
            ], [
                2.27289124e+02, 9.85998383e+01, 2.50334351e+02, 1.07137215e+02
            ], [
                1.86885681e+02, 1.07319916e+02, 3.00068634e+02, 1.52513535e+02
            ], [
                3.72980072e+02, 1.35389236e+02, 4.35769928e+02, 1.87310638e+02
            ], [
                5.99090942e+02, 1.05675484e+02, 6.27245361e+02, 1.21630264e+02
            ], [
                8.07875061e+01, 8.88861618e+01, 1.03188744e+02, 9.98524475e+01
            ], [
                6.11663574e+02, 1.09557632e+02, 6.40036987e+02, 1.35730301e+02
            ], [
                2.20839096e+02, 9.64170837e+01, 2.44063171e+02, 1.05758438e+02
            ], [
                4.82162292e+02, 1.08225266e+02, 5.23820923e+02, 1.28839401e+02
            ], [
                2.94147125e+02, 1.14885368e+02, 3.77608887e+02, 1.48902069e+02
            ], [
                7.77590027e+01, 8.83408508e+01, 9.87373352e+01, 9.83570938e+01
            ], [
                3.74932281e+02, 1.17987305e+02, 3.85349854e+02, 1.32233002e+02
            ], [
                9.76299438e+01, 8.96811218e+01, 1.17957596e+02, 1.00693565e+02
            ], [
                -4.88114357e-02, 1.09487862e+02, 6.19157715e+01, 1.43693024e+02
            ], [
                9.02799377e+01, 8.89016647e+01, 1.12263451e+02, 9.97362976e+01
            ], [
                5.91086670e+02, 1.08765915e+02, 6.18391479e+02, 1.24878296e+02
            ], [
                2.67454742e+02, 1.05075043e+02, 3.25762512e+02, 1.28307388e+02
            ], [
                1.35589050e+02, 9.19445801e+01, 1.59663986e+02, 1.03347069e+02
            ], [
                4.33314941e+02, 1.03436401e+02, 4.85610382e+02, 1.30874969e+02
            ]]),
            decimal=1)

    def test_dab_detr(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dab_detr/dab_detr_epoch_50.pth'
        config_path = 'configs/detection/dab_detr/dab_detr_r50_8x2_50e_coco.py'
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
        self.assertEqual(len(output['detection_boxes']), 14)
        self.assertEqual(len(output['detection_scores']), 14)
        self.assertEqual(len(output['detection_classes']), 14)

        assert_array_almost_equal(
            output['detection_classes'].tolist(),
            np.array([2, 2, 13, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                     dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([
                0.76882976, 0.7646885, 0.7161126, 0.690265, 0.66343737,
                0.6523155, 0.6338446, 0.6229081, 0.61183584, 0.56314564,
                0.5553375, 0.52696437, 0.5121799, 0.50143206
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
            np.array([[
                2.94298431e+02, 1.16071609e+02, 3.80441406e+02, 1.49636551e+02
            ], [
                4.80061157e+02, 1.09993500e+02, 5.23231689e+02, 1.30263199e+02
            ], [
                2.20323151e+02, 1.76510803e+02, 4.56516602e+02, 3.86306091e+02
            ], [
                1.67692703e+02, 1.08259041e+02, 2.14938675e+02, 1.38943848e+02
            ], [
                3.98115051e+02, 1.11344788e+02, 4.33720520e+02, 1.33362991e+02
            ], [
                4.30487427e+02, 1.04401749e+02, 4.84147034e+02, 1.32188812e+02
            ], [
                6.07396790e+02, 1.11725601e+02, 6.37299011e+02, 1.36237335e+02
            ], [
                2.67433319e+02, 1.05939735e+02, 3.27193970e+02, 1.30185196e+02
            ], [
                5.89790527e+02, 1.10369667e+02, 6.18799927e+02, 1.26195084e+02
            ], [
                3.37562561e-01, 1.10911972e+02, 6.30030289e+01, 1.46092499e+02
            ], [
                1.90680939e+02, 1.09017525e+02, 2.98907837e+02, 1.55803345e+02
            ], [
                5.67942505e+02, 1.10472374e+02, 5.94191406e+02, 1.27068993e+02
            ], [
                1.39744949e+02, 9.47335892e+01, 1.62575592e+02, 1.05453819e+02
            ], [
                6.21154976e+01, 9.22676468e+01, 8.40625458e+01, 1.04883873e+02
            ]]),
            decimal=1)

    def test_dn_detr(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dn_detr/dn_detr_epoch_50.pth'
        config_path = 'configs/detection/dab_detr/dn_detr_r50_8x2_50e_coco.py'
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
        self.assertEqual(len(output['detection_boxes']), 16)
        self.assertEqual(len(output['detection_scores']), 16)
        self.assertEqual(len(output['detection_classes']), 16)

        assert_array_almost_equal(
            output['detection_classes'].tolist(),
            np.array([2, 13, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 56],
                     dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([
                0.8800604, 0.8667884, 0.86659354, 0.80306965, 0.7643116,
                0.73749566, 0.72706455, 0.67101157, 0.63163954, 0.61646515,
                0.5724492, 0.55362254, 0.5403437, 0.515215, 0.5129325,
                0.5115242
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
            np.array([[
                2.94934113e+02, 1.15754349e+02, 3.77551117e+02, 1.50592712e+02
            ], [
                2.20573959e+02, 1.75971313e+02, 4.56900696e+02, 3.83259552e+02
            ], [
                4.79592896e+02, 1.09940025e+02, 5.23734253e+02, 1.30806046e+02
            ], [
                3.98695374e+02, 1.11459778e+02, 4.34043640e+02, 1.34191025e+02
            ], [
                1.66982147e+02, 1.09447891e+02, 2.10353058e+02, 1.39974823e+02
            ], [
                6.09432617e+02, 1.13080711e+02, 6.35908508e+02, 1.36743851e+02
            ], [
                2.68071960e+02, 1.05007935e+02, 3.27403564e+02, 1.28014572e+02
            ], [
                1.90774857e+02, 1.07428474e+02, 2.98357330e+02, 1.56284973e+02
            ], [
                5.91029602e+02, 1.10539055e+02, 6.20702881e+02, 1.27421104e+02
            ], [
                4.31661011e+02, 1.05048080e+02, 4.84486694e+02, 1.32458572e+02
            ], [
                5.96618652e-02, 1.11379456e+02, 6.29082794e+01, 1.44083389e+02
            ], [
                6.05408134e+01, 9.26343765e+01, 8.31398087e+01, 1.05740341e+02
            ], [
                5.69148499e+02, 1.10713043e+02, 5.95078918e+02, 1.27627998e+02
            ], [
                1.00577385e+02, 9.03523636e+01, 1.17681740e+02, 1.01768692e+02
            ], [
                1.40064575e+02, 9.42549286e+01, 1.61879669e+02, 1.04935501e+02
            ], [
                3.71020813e+02, 1.34599655e+02, 4.33997437e+02, 1.88007019e+02
            ]]),
            decimal=1)

    # def test_dino(self):
    #     model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dino/dino_4sc_r50_36e/epoch_29.pth'
    #     config_path = 'configs/detection/dino/dino_4sc_r50_36e_coco.py'
    #     img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
    #     model = DetectionPredictor(model_path, config_path)
    #     output = model(img)[0]
    #     with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
    #         tmp_save_path = tmp_file.name
    #         model.visualize(img, output, out_file=tmp_save_path)

    #     self.assertIn('detection_boxes', output)
    #     self.assertIn('detection_scores', output)
    #     self.assertIn('detection_classes', output)
    #     self.assertIn('img_metas', output)
    #     self.assertEqual(len(output['detection_boxes']), 300)
    #     self.assertEqual(len(output['detection_scores']), 300)
    #     self.assertEqual(len(output['detection_classes']), 300)

    #     assert_array_almost_equal(
    #         output['detection_classes'].tolist(),
    #         np.array([13, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32).tolist())

    #     assert_array_almost_equal(
    #         output['detection_scores'],
    #         np.array([
    #             0.8808171153068542, 0.8584598898887634, 0.8214247226715088,
    #             0.8156911134719849, 0.7707086801528931, 0.6717984080314636,
    #             0.6578451991081238, 0.6269607543945312, 0.6063129901885986,
    #             0.5223093628883362
    #         ],
    #                  dtype=np.float32),
    #         decimal=2)

    #     assert_array_almost_equal(
    #         output['detection_boxes'],
    #         np.array([[
    #             222.15492248535156, 175.9025421142578, 456.3177490234375,
    #             382.48211669921875
    #         ],
    #                   [
    #                       295.12115478515625, 115.97019958496094,
    #                       378.97119140625, 150.2149658203125
    #                   ],
    #                   [
    #                       190.94241333007812, 108.94568634033203,
    #                       298.280517578125, 155.6221160888672
    #                   ],
    #                   [
    #                       167.8346405029297, 109.49150085449219,
    #                       211.50537109375, 140.08895874023438
    #                   ],
    #                   [
    #                       482.0719909667969, 110.47320556640625,
    #                       523.1851806640625, 130.19410705566406
    #                   ],
    #                   [
    #                       609.3395385742188, 113.26068115234375,
    #                       635.8460083007812, 136.93771362304688
    #                   ],
    #                   [
    #                       266.5657958984375, 105.04171752929688,
    #                       326.9735107421875, 127.39012145996094
    #                   ],
    #                   [
    #                       431.43096923828125, 105.18028259277344,
    #                       484.13787841796875, 131.9821319580078
    #                   ],
    #                   [
    #                       60.43342971801758, 94.02497100830078,
    #                       86.346435546875, 106.31623840332031
    #                   ],
    #                   [
    #                       139.32015991210938, 96.0668716430664,
    #                       167.1505126953125, 105.44377899169922
    #                   ]]),
    #         decimal=1)


if __name__ == '__main__':
    unittest.main()
