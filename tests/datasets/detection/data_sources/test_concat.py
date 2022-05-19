# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

from tests.ut_config import (DET_DATA_RAW_LOCAL, DET_DATA_SMALL_VOC_LOCAL,
                             VOC_CLASSES)

from easycv.datasets.builder import build_datasource


class DetSourceCocoTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_concat_source(self):
        data_root = DET_DATA_RAW_LOCAL
        data_source = dict(
            type='SourceConcat',
            data_source_list=[
                dict(
                    type='DetSourceRaw',
                    img_root_path=os.path.join(data_root, 'images/train2017'),
                    label_root_path=os.path.join(data_root,
                                                 'labels/train2017')),
                dict(
                    type='DetSourceRaw',
                    img_root_path=os.path.join(data_root, 'images/train2017'),
                    label_root_path=os.path.join(data_root,
                                                 'labels/train2017'))
            ])

        data_source = build_datasource(data_source)
        index_list = random.choices(list(range(20)), k=3)
        for idx in index_list:
            data = data_source.get_sample(idx)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertIn('filename', data)
            self.assertIn('gt_labels', data)
            self.assertEqual(data['img'].shape[-1], 3)
            self.assertEqual(len(data['img_shape']), 3)

        length = data_source.get_length()
        self.assertEqual(length, 256)

    def test_concat_diff_source(self):
        raw_data_root = DET_DATA_RAW_LOCAL
        voc_data_root = DET_DATA_SMALL_VOC_LOCAL
        data_source = dict(
            type='SourceConcat',
            data_source_list=[
                dict(
                    type='DetSourceVOC',
                    path=os.path.join(voc_data_root,
                                      'ImageSets/Main/train_20.txt'),
                    classes=VOC_CLASSES),
                dict(
                    type='DetSourceRaw',
                    img_root_path=os.path.join(raw_data_root,
                                               'images/train2017'),
                    label_root_path=os.path.join(raw_data_root,
                                                 'labels/train2017'))
            ])

        data_source = build_datasource(data_source)
        index_list = random.choices(list(range(20)), k=3)
        for idx in index_list:
            data = data_source.get_sample(idx)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertIn('filename', data)
            self.assertIn('gt_labels', data)
            self.assertEqual(data['img'].shape[-1], 3)
            self.assertEqual(len(data['img_shape']), 3)

        length = data_source.get_length()
        self.assertEqual(length, 148)


if __name__ == '__main__':
    unittest.main()
