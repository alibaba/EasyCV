# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from tests.ut_config import (IMG_NORM_CFG_255, SEG_DATA_SMALL_RAW_LOCAL,
                             VOC_CLASSES)

from easycv.core.evaluation.builder import build_evaluator
from easycv.datasets.builder import build_datasource
from easycv.datasets.segmentation.data_sources.raw import SegSourceRaw
from easycv.datasets.segmentation.raw import SegDataset
from easycv.file import io


class SegDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        data_root = SEG_DATA_SMALL_RAW_LOCAL
        data_source_cfg = dict(
            type='SegSourceRaw',
            img_root=os.path.join(data_root, 'images'),
            label_root=os.path.join(data_root, 'labels'),
            classes=VOC_CLASSES,
            num_processes=
            1  # results copy from datasource, ensure results and groundtruth has the same data list
        )
        crop_size = (512, 512)
        pipeline = [
            dict(
                type='SegRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='MMNormalize', **IMG_NORM_CFG_255),
            dict(type='MMPad', size=crop_size, pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg']),
        ]

        dataset = SegDataset(data_source_cfg, pipeline)
        data_source = build_datasource(data_source_cfg)
        gt_seg_maps = []
        for i in range(len(data_source)):
            sample = data_source.get_sample(i)
            gt_seg_maps.append(sample['gt_semantic_seg'])
        results = {'seg_pred': gt_seg_maps}

        evaluator = build_evaluator(
            dict(
                type='SegmentationEvaluator',
                classes=VOC_CLASSES,
                metric_names=['mIoU'],
            ))
        eval_results = dataset.evaluate(results, evaluators=evaluator)
        self.assertEqual(eval_results['aAcc'], 1.0)
        self.assertEqual(eval_results['mIoU'], 1.0)
        self.assertEqual(eval_results['mAcc'], 1.0)
        self.assertEqual(eval_results['IoU.aeroplane'], 1.0)
        self.assertEqual(eval_results['IoU.tvmonitor'], 1.0)
        self.assertEqual(eval_results['Acc.tvmonitor'], 1.0)


if __name__ == '__main__':
    unittest.main()
