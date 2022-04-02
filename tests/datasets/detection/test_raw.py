# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest

from tests.ut_config import DET_DATA_RAW_LOCAL, IMG_NORM_CFG_255

from easycv.datasets.detection import DetDataset


class DetDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_load(self):
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


if __name__ == '__main__':
    unittest.main()
