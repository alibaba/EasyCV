# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

from tests.ut_config import SEG_DATA_SAMLL_CITYSCAPES

from easycv.datasets.segmentation.data_sources.cityscapes import \
    SegSourceCityscapes

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')


class SegSourceCityscapesTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_cityscapes(self):

        data_source = SegSourceCityscapes(
            img_root=os.path.join(SEG_DATA_SAMLL_CITYSCAPES, 'leftImg8bit'),
            label_root=os.path.join(SEG_DATA_SAMLL_CITYSCAPES, 'gtFine'),
            classes=CLASSES,
        )

        index_list = random.choices(list(range(20)), k=3)
        for idx in index_list:
            data = data_source[idx]
            self.assertIn('img_fields', data)
            self.assertIn('seg_fields', data)


if __name__ == '__main__':
    unittest.main()
