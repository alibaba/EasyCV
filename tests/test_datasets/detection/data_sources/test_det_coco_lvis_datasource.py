# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

import numpy as np
from tests.ut_config import COCO_CLASSES, DET_DATASET_DOWNLOAD_SMALL

from easycv.datasets.builder import build_datasource


class DetSourceCocoLvis(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source):
        index_list = random.choices(list(range(20)), k=3)

        for idx in index_list:
            data = data_source[idx]

            self.assertIn('ann_info', data)
            self.assertIn('img_info', data)
            self.assertIn('filename', data)
            self.assertEqual(data['img'].shape[-1], 3)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertGreater(len(data['gt_labels']), 0)

        length = len(data_source)

        self.assertEqual(length, 20)

        exists = False
        for idx in range(length):

            result = data_source[idx]

            file_name = result.get('filename', '')

            if file_name.endswith('000000290676.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (427, 640, 3))
                self.assertEqual(
                    result['gt_labels'].tolist(),
                    np.array([34, 34, 34, 34, 34, 34, 31, 35, 26],
                             dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].tolist(),
                    np.array([[
                        444.2699890136719, 215.5, 557.010009765625,
                        328.20001220703125
                    ],
                              [
                                  343.3900146484375, 316.760009765625,
                                  392.6099853515625, 352.3900146484375
                              ], [0.0, 0.0, 464.1099853515625, 427.0],
                              [
                                  329.82000732421875, 320.32000732421875,
                                  342.94000244140625, 347.94000244140625
                              ],
                              [
                                  319.32000732421875, 343.1600036621094,
                                  342.6199951171875, 363.0899963378906
                              ],
                              [
                                  363.7099914550781, 302.010009765625,
                                  383.07000732421875, 315.1300048828125
                              ],
                              [
                                  413.260009765625, 371.82000732421875,
                                  507.30999755859375, 390.69000244140625
                              ],
                              [
                                  484.0400085449219, 322.0, 612.47998046875,
                                  422.510009765625
                              ],
                              [
                                  393.79998779296875, 287.9599914550781,
                                  497.6000061035156, 377.4800109863281
                              ]],
                             dtype=np.float32).tolist())
                break

        self.assertTrue(exists)

    def test_download_coco_lvis(self):
        pipeline = [
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ]

        cfg = dict(
            links=[
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/samll_lvis/lvis_v1_small_train.json.zip',
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/samll_lvis/lvis_v1_small_val.json.zip',
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/samll_lvis/train2017.zip',
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/samll_lvis/val2017.zip'
            ],
            train='lvis_v1_small_train.json',
            val='lvis_v1_small_train.json',
            dataset='images'
            # default
        )

        datasource_cfg = dict(
            type='DetSourceLvis',
            pipeline=pipeline,
            path=DET_DATASET_DOWNLOAD_SMALL,
            classes=COCO_CLASSES,
            split='train',
            download=True,
            cfg=cfg)
        data_source = build_datasource(datasource_cfg)
        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
