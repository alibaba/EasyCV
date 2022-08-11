# Copyright (c) Alibaba, Inc. and its affiliates.
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
        detr = DetectionPredictor(model_path, config_path)
        output = detr.predict(img)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 100)
        self.assertEqual(len(output['detection_scores'][0]), 100)
        self.assertEqual(len(output['detection_classes'][0]), 100)

        self.assertListEqual(
            output['detection_classes'][0].tolist(),
            np.array([
                2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 2, 13, 2, 0, 13, 2,
                0, 2, 56, 2, 7, 2, 2, 2, 2, 2, 2, 2, 7, 56, 2, 2, 7, 7, 2, 7,
                2, 2, 56, 2, 7, 11, 2, 2, 2, 0, 7, 2, 2, 2, 2, 2, 7, 2, 2, 7,
                2, 2, 2, 2, 13, 2, 2, 2, 13, 2, 2, 56, 2, 56, 2, 7, 56, 13, 7,
                56, 2, 0, 2, 7, 2, 7, 2, 56, 2, 2, 2, 7, 56, 2, 2, 7, 2, 0, 2,
                2
            ],
                     dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0],
            np.array([
                0.07836595922708511, 0.219977006316185, 0.5831383466720581,
                0.4256463646888733, 0.9853266477584839, 0.24607707560062408,
                0.28005731105804443, 0.500579833984375, 0.09835881739854813,
                0.05178987979888916, 0.07282666862010956, 0.9802166819572449,
                0.04826607555150986, 0.06967002153396606, 0.9541336894035339,
                0.36800140142440796, 0.2821184992790222, 0.009824174456298351,
                0.981455385684967, 0.9890823364257812, 0.11702633649110794,
                0.3397829532623291, 0.03982163220643997, 0.06306332349777222,
                0.07951728254556656, 0.949343204498291, 0.1537322700023651,
                0.3483341634273529, 0.044335901737213135, 0.03239326551556587,
                0.11274639517068863, 0.462695449590683, 0.03906852751970291,
                0.006577627267688513, 0.651928722858429, 0.13711832463741302,
                0.15317879617214203, 0.5399832129478455, 0.08868053555488586,
                0.026992695406079292, 0.0887782946228981, 0.081451416015625,
                0.5485899448394775, 0.1959853619337082, 0.20348815619945526,
                0.1804366111755371, 0.04546552523970604, 0.4005874693393707,
                0.4241448938846588, 0.20359477400779724, 0.18858052790164948,
                0.5971255898475647, 0.6823391914367676, 0.09363959729671478,
                0.9855959415435791, 0.5903261303901672, 0.0731084868311882,
                0.9813686609268188, 0.9814890027046204, 0.11285952478647232,
                0.46758928894996643, 0.5004158616065979, 0.5852540731430054,
                0.1944422572851181, 0.04896926134824753, 0.17205820977687836,
                0.188123881816864, 0.43242165446281433, 0.3784835636615753,
                0.06754120439291, 0.8264386057853699, 0.054902296513319016,
                0.05457871034741402, 0.05988362058997154, 0.054624997079372406,
                0.37744957208633423, 0.08150151371955872, 0.015097505412995815,
                0.1074686348438263, 0.004187499638646841, 0.9733405709266663,
                0.15225540101528168, 0.711842954158783, 0.06490222364664078,
                0.9512462615966797, 0.03674759343266487, 0.09688679873943329,
                0.02119528315961361, 0.9736435413360596, 0.9338251948356628,
                0.09611554443836212, 0.09142979979515076, 0.01647237129509449,
                0.9805111289024353, 0.3779929280281067, 0.09553579986095428,
                0.11411411315202713, 0.0063759335316717625, 0.2972108721733093,
                0.07761078327894211
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0],
            np.array([[
                131.10389709472656, 90.93302154541016, 148.95504760742188,
                101.69216918945312
            ],
                      [
                          239.10910034179688, 113.36551666259766,
                          256.0523376464844, 125.22894287109375
                      ],
                      [
                          132.1316375732422, 90.8366470336914,
                          151.00839233398438, 101.83119201660156
                      ],
                      [
                          579.37646484375, 108.26667785644531,
                          605.0717163085938, 124.79525756835938
                      ],
                      [
                          189.69073486328125, 108.04875946044922,
                          296.8011779785156, 154.44204711914062
                      ],
                      [
                          588.5413208007812, 107.89535522460938,
                          615.6463012695312, 124.41362762451172
                      ],
                      [
                          57.38536071777344, 89.7335433959961,
                          79.20274353027344, 102.61941528320312
                      ],
                      [
                          163.97628784179688, 92.95049285888672,
                          180.87033081054688, 102.6163330078125
                      ],
                      [
                          127.82454681396484, 90.27918243408203,
                          144.6781768798828, 99.71304321289062
                      ],
                      [
                          438.4545593261719, 103.00477600097656,
                          480.4275817871094, 121.69993591308594
                      ],
                      [
                          23.33353042602539, 113.83180236816406,
                          58.04608154296875, 141.02174377441406
                      ],
                      [
                          165.77127075195312, 108.23619842529297,
                          208.61306762695312, 136.4344482421875
                      ],
                      [
                          402.93792724609375, 110.03141784667969,
                          437.510009765625, 132.2660369873047
                      ],
                      [
                          571.1912841796875, 108.44816589355469,
                          596.1377563476562, 125.31720733642578
                      ],
                      [
                          564.8051147460938, 108.13015747070312,
                          593.9150390625, 126.26905822753906
                      ],
                      [
                          79.12519836425781, 105.58458709716797,
                          111.23175048828125, 119.07565307617188
                      ],
                      [
                          549.2890625, 110.77001190185547, 563.7803955078125,
                          122.94622039794922
                      ],
                      [
                          384.75567626953125, 118.3704605102539,
                          422.6398010253906, 138.06492614746094
                      ],
                      [
                          218.92532348632812, 177.14031982421875,
                          459.10760498046875, 381.1133728027344
                      ],
                      [
                          397.3675231933594, 110.41165161132812,
                          436.5208740234375, 133.16848754882812
                      ],
                      [
                          239.1329803466797, 114.10742950439453,
                          255.9464874267578, 126.29271697998047
                      ],
                      [
                          214.19888305664062, 95.95294952392578,
                          233.9381103515625, 104.9795913696289
                      ],
                      [
                          395.9658508300781, 148.40223693847656,
                          434.2126770019531, 182.97567749023438
                      ],
                      [
                          269.0050964355469, 104.73245239257812,
                          320.3499755859375, 123.80233001708984
                      ],
                      [
                          483.2522888183594, 107.74896240234375, 522.5078125,
                          128.4845428466797
                      ],
                      [
                          57.623504638671875, 90.50349426269531,
                          82.20435333251953, 103.5736312866211
                      ],
                      [
                          375.03070068359375, 117.73094940185547,
                          385.3664855957031, 132.47479248046875
                      ],
                      [
                          555.9656372070312, 102.33853912353516,
                          568.1951293945312, 113.82102966308594
                      ],
                      [
                          388.681640625, 107.18630981445312,
                          415.21905517578125, 121.56800842285156
                      ],
                      [
                          510.5431823730469, 107.02037811279297,
                          533.8384399414062, 122.61180114746094
                      ],
                      [
                          187.148193359375, 100.21558380126953,
                          253.47540283203125, 123.25538635253906
                      ],
                      [
                          552.3801879882812, 103.33021545410156,
                          564.61865234375, 115.99454498291016
                      ],
                      [
                          425.8926086425781, 104.35319519042969,
                          477.8686218261719, 130.82357788085938
                      ],
                      [
                          222.24378967285156, 176.4434051513672,
                          456.42266845703125, 312.5479431152344
                      ],
                      [
                          227.29019165039062, 98.5999755859375,
                          250.33477783203125, 107.1373291015625
                      ],
                      [
                          165.81600952148438, 107.86138916015625,
                          202.11196899414062, 134.08160400390625
                      ],
                      [
                          175.83389282226562, 89.20259857177734,
                          224.58187866210938, 105.41484832763672
                      ],
                      [
                          186.885986328125, 107.32003021240234,
                          300.068115234375, 152.51370239257812
                      ],
                      [
                          165.5398712158203, 107.98709106445312,
                          202.61941528320312, 134.54295349121094
                      ],
                      [
                          611.3699951171875, 110.01651000976562, 639.626953125,
                          133.97329711914062
                      ],
                      [
                          550.5084838867188, 104.33821105957031,
                          562.5010986328125, 115.29130554199219
                      ],
                      [
                          59.68817901611328, 97.86705017089844,
                          76.69844055175781, 110.52032470703125
                      ],
                      [
                          372.9800720214844, 135.38967895507812,
                          435.77044677734375, 187.3105010986328
                      ],
                      [
                          233.5430908203125, 99.0816421508789,
                          255.1123809814453, 109.31993103027344
                      ],
                      [
                          57.52912521362305, 90.85675048828125,
                          81.06048583984375, 104.28386688232422
                      ],
                      [
                          566.7468872070312, 81.79965209960938,
                          581.5723266601562, 92.56966400146484
                      ],
                      [
                          166.86119079589844, 108.41394805908203,
                          205.91815185546875, 135.9706268310547
                      ],
                      [
                          87.78324890136719, 89.354736328125,
                          108.06605529785156, 99.89331817626953
                      ],
                      [
                          -0.0010925531387329102, 111.63217163085938,
                          13.02929401397705, 123.92339324951172
                      ],
                      [
                          235.19432067871094, 114.09554290771484,
                          251.94717407226562, 126.1430892944336
                      ],
                      [
                          268.6304626464844, 104.09455108642578,
                          328.7283935546875, 124.19095611572266
                      ],
                      [
                          599.0895385742188, 105.6767807006836, 627.2431640625,
                          121.63172912597656
                      ],
                      [
                          80.78914642333984, 88.88621520996094,
                          103.19034576416016, 99.85254669189453
                      ],
                      [
                          620.0072021484375, 109.53975677490234,
                          640.0657958984375, 133.46539306640625
                      ],
                      [
                          611.6638793945312, 109.55789947509766,
                          640.0369873046875, 135.73045349121094
                      ],
                      [
                          220.8399200439453, 96.41732025146484,
                          244.06399536132812, 105.75860595703125
                      ],
                      [
                          434.6024169921875, 105.29331970214844,
                          482.67218017578125, 130.61903381347656
                      ],
                      [
                          482.16302490234375, 108.22539520263672,
                          523.8209228515625, 128.8394317626953
                      ],
                      [
                          294.1483154296875, 114.8856201171875,
                          377.6090087890625, 148.9021453857422
                      ],
                      [
                          197.5174560546875, 91.83529663085938,
                          224.16571044921875, 104.21776580810547
                      ],
                      [
                          167.1007080078125, 94.26935577392578,
                          185.2867431640625, 103.67475128173828
                      ],
                      [
                          77.7591552734375, 88.3407974243164,
                          98.73750305175781, 98.35700988769531
                      ],
                      [
                          374.9325866699219, 117.9875259399414,
                          385.34991455078125, 132.2331085205078
                      ],
                      [
                          167.0509490966797, 108.91958618164062,
                          205.8968505859375, 136.49874877929688
                      ],
                      [
                          51.54475784301758, 104.56134796142578,
                          73.30614471435547, 120.99707794189453
                      ],
                      [
                          274.94195556640625, 101.94827270507812,
                          323.97650146484375, 115.7809829711914
                      ],
                      [
                          236.10757446289062, 97.75923919677734,
                          254.6915283203125, 107.15653228759766
                      ],
                      [
                          609.9969482421875, 100.36739349365234,
                          638.1365966796875, 115.2613754272461
                      ],
                      [
                          74.30718994140625, 103.82154083251953,
                          108.9682388305664, 118.71984100341797
                      ],
                      [
                          367.69842529296875, 118.2647933959961,
                          380.7364501953125, 132.5258331298828
                      ],
                      [
                          97.63047790527344, 89.68116760253906,
                          117.95793151855469, 100.69364166259766
                      ],
                      [
                          373.5556640625, 135.07652282714844,
                          434.33709716796875, 185.4060516357422
                      ],
                      [
                          54.454566955566406, 104.98534393310547,
                          71.63267517089844, 119.91905212402344
                      ],
                      [
                          375.6446838378906, 135.28944396972656,
                          435.38543701171875, 185.16932678222656
                      ],
                      [
                          614.5422973632812, 107.62055969238281,
                          640.0924072265625, 132.63742065429688
                      ],
                      [
                          433.181396484375, 103.48396301269531,
                          484.539794921875, 131.49851989746094
                      ],
                      [
                          375.8929443359375, 142.8272247314453,
                          434.06439208984375, 186.07620239257812
                      ],
                      [
                          331.8176574707031, 131.65638732910156,
                          437.4866027832031, 185.01356506347656
                      ],
                      [
                          0.02536296844482422, 109.71121978759766,
                          59.71095657348633, 143.3699951171875
                      ],
                      [
                          225.39547729492188, 179.25990295410156,
                          453.9075012207031, 378.88177490234375
                      ],
                      [
                          -0.048813819885253906, 109.48809051513672,
                          61.9161376953125, 143.69325256347656
                      ],
                      [
                          230.8810272216797, 113.71331787109375,
                          246.87635803222656, 125.97274780273438
                      ],
                      [
                          90.28150939941406, 88.90176391601562,
                          112.26461791992188, 99.73652648925781
                      ],
                      [
                          398.63372802734375, 109.19901275634766,
                          437.08404541015625, 131.7988739013672
                      ],
                      [
                          591.0872802734375, 108.76611328125,
                          618.3915405273438, 124.87841796875
                      ],
                      [
                          294.1384582519531, 114.44403076171875,
                          380.4442138671875, 149.1887664794922
                      ],
                      [
                          203.1075439453125, 109.37834167480469,
                          235.15982055664062, 125.3521957397461
                      ],
                      [
                          223.9816436767578, 177.92266845703125,
                          456.14263916015625, 357.9170837402344
                      ],
                      [
                          267.4551086425781, 105.07503509521484,
                          325.762939453125, 128.3075408935547
                      ],
                      [
                          135.58905029296875, 91.94464111328125,
                          159.6639404296875, 103.34713745117188
                      ],
                      [
                          540.2098388671875, 103.45868682861328,
                          555.146240234375, 116.43596649169922
                      ],
                      [
                          293.979736328125, 114.60274505615234,
                          380.0762939453125, 149.8252716064453
                      ],
                      [
                          220.00042724609375, 175.55282592773438,
                          452.283447265625, 327.056640625
                      ],
                      [
                          433.3155822753906, 103.43656158447266,
                          485.6103515625, 130.87503051757812
                      ],
                      [
                          553.2203369140625, 101.77803802490234,
                          564.004150390625, 112.206298828125
                      ],
                      [
                          567.9517822265625, 107.1722640991211,
                          595.299560546875, 124.94164276123047
                      ],
                      [
                          555.1934814453125, 109.10118103027344,
                          572.1039428710938, 122.53047180175781
                      ],
                      [
                          77.2689208984375, 90.0588607788086,
                          501.6678466796875, 346.69378662109375
                      ],
                      [
                          552.4683227539062, 111.0732650756836,
                          567.466064453125, 123.53128814697266
                      ],
                      [
                          79.25263977050781, 89.3648452758789,
                          111.41500854492188, 101.7647476196289
                      ]]),
            decimal=1)

    def test_dab_detr(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dab_detr/dab_detr_epoch_50.pth'
        config_path = 'configs/detection/dab_detr/dab_detr_r50_8x2_50e_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        dab_detr = DetectionPredictor(model_path, config_path)
        output = dab_detr.predict(img)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 10)
        self.assertEqual(len(output['detection_scores'][0]), 10)
        self.assertEqual(len(output['detection_classes'][0]), 10)

        self.assertListEqual(
            output['detection_classes'][0].tolist(),
            np.array([2, 2, 13, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0],
            np.array([
                0.7688284516334534, 0.7646799683570862, 0.7159939408302307,
                0.6902833580970764, 0.6633996367454529, 0.6523147821426392,
                0.633848249912262, 0.6229104995727539, 0.611840009689331,
                0.5631589293479919
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0],
            np.array([[
                294.2984313964844, 116.07160949707031, 380.4406433105469,
                149.6365509033203
            ],
                      [
                          480.0610656738281, 109.99347686767578,
                          523.2314453125, 130.26318359375
                      ],
                      [
                          220.32269287109375, 176.51010131835938,
                          456.51715087890625, 386.30767822265625
                      ],
                      [
                          167.6925506591797, 108.25935363769531,
                          214.93780517578125, 138.94424438476562
                      ],
                      [
                          398.1152648925781, 111.34457397460938,
                          433.72052001953125, 133.36280822753906
                      ],
                      [
                          430.48736572265625, 104.4018325805664,
                          484.1470947265625, 132.18893432617188
                      ],
                      [
                          607.396728515625, 111.72560119628906,
                          637.2987670898438, 136.2375946044922
                      ],
                      [
                          267.43353271484375, 105.93965911865234,
                          327.1937561035156, 130.18527221679688
                      ],
                      [
                          589.790771484375, 110.36975860595703,
                          618.8001098632812, 126.1950454711914
                      ],
                      [
                          0.3374290466308594, 110.91182708740234,
                          63.00359344482422, 146.0926971435547
                      ]]),
            decimal=1)

    def test_dn_detr(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dn_detr/dn_detr_epoch_50.pth'
        config_path = 'configs/detection/dab_detr/dn_detr_r50_8x2_50e_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        dn_detr = DetectionPredictor(model_path, config_path)
        output = dn_detr.predict(img)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 10)
        self.assertEqual(len(output['detection_scores'][0]), 10)
        self.assertEqual(len(output['detection_classes'][0]), 10)

        self.assertListEqual(
            output['detection_classes'][0].tolist(),
            np.array([2, 13, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0],
            np.array([
                0.8800525665283203, 0.866659939289093, 0.8665854930877686,
                0.8030595183372498, 0.7642921209335327, 0.7375038862228394,
                0.7270554304122925, 0.6710091233253479, 0.6316548585891724,
                0.6164721846580505
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0],
            np.array([[
                294.9338073730469, 115.7542495727539, 377.5517578125,
                150.59274291992188
            ],
                      [
                          220.57424926757812, 175.97023010253906,
                          456.9001770019531, 383.2597351074219
                      ],
                      [
                          479.5928649902344, 109.94012451171875,
                          523.7343139648438, 130.80604553222656
                      ],
                      [
                          398.6956787109375, 111.45973205566406,
                          434.0437316894531, 134.1909637451172
                      ],
                      [
                          166.98208618164062, 109.44792938232422,
                          210.35342407226562, 139.9746856689453
                      ],
                      [
                          609.432373046875, 113.08062744140625,
                          635.9082641601562, 136.74383544921875
                      ],
                      [
                          268.0716552734375, 105.00788879394531,
                          327.4037170410156, 128.01449584960938
                      ],
                      [
                          190.77467346191406, 107.42850494384766,
                          298.35760498046875, 156.2850341796875
                      ],
                      [
                          591.0296020507812, 110.53913116455078,
                          620.702880859375, 127.42123413085938
                      ],
                      [
                          431.6607971191406, 105.04813385009766,
                          484.4869689941406, 132.45864868164062
                      ]]),
            decimal=1)


if __name__ == '__main__':
    unittest.main()
