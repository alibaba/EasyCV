# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest

import numpy as np
from tests.ut_config import DET_DATA_RAW_LOCAL, IMG_NORM_CFG_255

from easycv.datasets.detection import DetDataset


class DetDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _get_dataset(self):
        img_scale = (640, 640)
        data_source_cfg = dict(
            type='DetSourceRaw',
            img_root_path=os.path.join(DET_DATA_RAW_LOCAL, 'images/train2017'),
            label_root_path=os.path.join(DET_DATA_RAW_LOCAL,
                                         'labels/train2017'))

        pipeline = [
            dict(type='MMResize', img_scale=img_scale, keep_ratio=True),
            dict(
                type='MMPad',
                pad_to_square=True,
                pad_val=(114.0, 114.0, 114.0)),
            dict(type='MMNormalize', **IMG_NORM_CFG_255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]

        dataset = DetDataset(data_source=data_source_cfg, pipeline=pipeline)

        return dataset

    def test_load(self):
        dataset = self._get_dataset()
        data_num = len(dataset)
        s = time.time()
        for data in dataset:
            pass
        t = time.time()
        print(f'read data done {(t-s)/data_num}s per sample')
        self.assertTrue('img' in data)
        self.assertTrue('gt_labels' in data)
        self.assertTrue('gt_bboxes' in data)
        self.assertTrue('img_metas' in data)
        img_metas = data['img_metas'].data
        self.assertTrue('img_shape' in img_metas)
        self.assertTrue('ori_img_shape' in img_metas)

    def test_visualize(self):
        dataset = self._get_dataset()
        count = 5
        detection_boxes = []
        detection_classes = []
        img_metas = []
        for i, data in enumerate(dataset):
            detection_boxes.append(
                data['gt_bboxes'].data.cpu().detach().numpy())
            detection_classes.append(
                data['gt_labels'].data.cpu().detach().numpy())
            img_metas.append(data['img_metas'].data)
            if i > count:
                break

        detection_scores = []
        for classes in detection_classes:
            detection_scores.append(0.1 * np.array(range(len(classes))))

        results = {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
            'img_metas': img_metas
        }
        output = dataset.visualize(results, vis_num=2, score_thr=0.1)
        self.assertEqual(len(output['images']), 2)
        self.assertEqual(len(output['img_metas']), 2)
        self.assertEqual(len(output['images'][0].shape), 3)
        self.assertIn('filename', output['img_metas'][0])


if __name__ == '__main__':
    unittest.main()
