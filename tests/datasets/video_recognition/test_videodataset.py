# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest


from tests.ut_config import VIDEO_DATA_SMALL_RAW_LOCAL
from easycv.core.evaluation.builder import build_evaluator
from easycv.datasets.builder import build_datasource
from easycv.datasets.video_recognition.raw import VideoDataset


class VideoDatasetTest(unittest.TestCase):
    
    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        
    
    def test_default(self):
        data_root = VIDEO_DATA_SMALL_RAW_LOCAL
        data_source_cfg = dict(
            type='VideoDatasource',
            ann_file = os.path.join(data_root, 'kinetics400/test.txt'),
            data_root = data_root,
            split = ' ', 
        )
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
        pipeline = [
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
            dict(type='DecordDecode'),
            dict(type='VideoResize', scale=(-1, 256)),
            dict(type='VideoRandomResizedCrop'),
            dict(type='VideoResize', scale=(224, 224), keep_ratio=False),
            dict(type='VideoFlip', flip_ratio=0.5),
            dict(type='VideoNormalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='VideoToTensor', keys=['imgs', 'label'])
        ]
        
        dataset = VideoDataset(data_source_cfg, pipeline)
        
        item = dataset[10]
        print(item.keys())
        print(item['imgs'].shape)
        print(item['label'])
        
    def test_video_text(self):
        data_root = VIDEO_DATA_SMALL_RAW_LOCAL
        data_source_cfg = dict(
            type='VideoTextDatasource',
            ann_file = os.path.join(data_root, 'video_text/test.txt'),
            data_root = data_root+'/video_text/video', 
        )
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
        pipeline = [
            dict(type='DecordInit'),
            dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
            dict(type='DecordDecode'),
            dict(type='VideoResize', scale=(-1, 256)),
            dict(type='VideoRandomResizedCrop'),
            dict(type='VideoResize', scale=(224, 224), keep_ratio=False),
            dict(type='VideoFlip', flip_ratio=0.5),
            dict(type='VideoNormalize', **img_norm_cfg),
            dict(type='TextTokenizer'),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label','text_input_ids','text_input_mask'], meta_keys=[]),
            dict(type='VideoToTensor', keys=['imgs', 'label'])
        ]
        
        dataset = VideoDataset(data_source_cfg, pipeline)
        
        item = dataset[5]
        print(item.keys())
        print(item['imgs'].dtype)
        print(item['label'])
        print(item['text_input_ids'])
        print(item['text_input_mask'])
        
        
if __name__ == '__main__':
    unittest.main()
        
        