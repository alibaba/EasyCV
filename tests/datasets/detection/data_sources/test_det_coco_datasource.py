# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

import numpy as np
from tests.ut_config import (COCO_CLASSES, COCO_DATASET_DOWNLOAD_SMALL,
                             DET_DATA_COCO2017_DOWNLOAD,
                             DET_DATA_SMALL_COCO_LOCAL)

from easycv.datasets.detection.data_sources.coco import (DetSourceCoco,
                                                         DetSourceCoco2017)
from easycv.datasets.utils.download_data.download_coco import download_coco


class DetSourceCocoTest(unittest.TestCase):

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
            self.assertGreater(len(data['gt_labels']), 1)

        length = len(data_source)

        self.assertEqual(length, 20)

        exists = False
        for idx in range(length):

            result = data_source[idx]

            file_name = result.get('filename', '')

            if file_name.endswith('000000224736.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (427, 640, 3))
                self.assertEqual(result['gt_labels'].tolist(),
                                 np.array([61, 71], dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].tolist(),
                    np.array([[148.1, 297.65, 270.24, 383.24],
                              [470.09, 148.13, 552.07, 207.29]],
                             dtype=np.float32).tolist())
                break

        self.assertTrue(exists)

    def test_det_source_coco(self):
        data_root = DET_DATA_SMALL_COCO_LOCAL

        data_source = DetSourceCoco(
            ann_file=os.path.join(data_root, 'instances_train2017_20.json'),
            img_prefix=os.path.join(data_root, 'train2017'),
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            classes=COCO_CLASSES,
            filter_empty_gt=False,
            iscrowd=False)

        self._base_test(data_source)

    def test_det_source_coco2017(self):

        data_root = DET_DATA_COCO2017_DOWNLOAD

        data_source = DetSourceCoco2017(
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            path=data_root,
            download=True,
            split='train',
            classes=COCO_CLASSES,
            filter_empty_gt=False,
            iscrowd=False)

        self._base_test(data_source)

    def test_download_coco(self):
        cfg = dict(
            coco2017=[
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/annotations.zip',
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/train2017_small.zip',
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/val2017_small.zip',
            ],
            detection=dict(
                train='instances_train2017.json',
                val='instances_val2017.json',
            ),
            train_dataset='train2017',
            val_dataset='val2017',
            pose=dict(
                train='person_keypoints_train2017.json',
                val='person_keypoints_val2017.json'))

        coco_path = download_coco(
            'coco2017',
            'train',
            target_dir=COCO_DATASET_DOWNLOAD_SMALL,
            cfg=cfg)
        data_source = DetSourceCoco(
            ann_file=coco_path['ann_file'],
            img_prefix=coco_path['img_prefix'],
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            classes=COCO_CLASSES,
            filter_empty_gt=False,
            iscrowd=False)
        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
