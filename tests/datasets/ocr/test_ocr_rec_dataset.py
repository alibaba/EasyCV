# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from tests.ut_config import SMALL_OCR_REC_DATA

from easycv.datasets.builder import build_dataset


class OCRRecsDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _get_dataset(self):
        data_root = SMALL_OCR_REC_DATA
        data_train_list = os.path.join(data_root, 'label.txt')
        pipeline = [
            dict(type='RecConAug', prob=0.5, image_shape=(48, 320, 3)),
            dict(type='RecAug'),
            dict(
                type='MultiLabelEncode',
                max_text_length=25,
                use_space_char=True,
                character_dict_path=
                'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/ocr/dict/ppocr_keys_v1.txt',
            ),
            dict(type='RecResizeImg', image_shape=(3, 48, 320)),
            dict(type='MMToTensor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'label_ctc', 'label_sar', 'length', 'valid_ratio'
                ],
                meta_keys=['img_path'])
        ]
        data = dict(
            train=dict(
                type='OCRRecDataset',
                data_source=dict(
                    type='OCRRecSource',
                    label_file=data_train_list,
                    data_dir=SMALL_OCR_REC_DATA + '/img',
                    ext_data_num=0,
                    test_mode=True,
                ),
                pipeline=pipeline))
        dataset = build_dataset(data['train'])

        return dataset

    def test_default(self):
        dataset = self._get_dataset()
        for _, batch in enumerate(dataset):

            img, label_ctc, label_sar = batch['img'], batch[
                'label_ctc'], batch['label_sar']
            self.assertEqual(img.shape, torch.Size([3, 48, 320]))
            self.assertEqual(label_ctc.shape, (25, ))
            self.assertEqual(label_sar.shape, (25, ))
            break


if __name__ == '__main__':
    unittest.main()
