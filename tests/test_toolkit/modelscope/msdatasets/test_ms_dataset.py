# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode


class MsDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_streaming_load_coco(self):
        small_coco_for_test = MsDataset.load(
            dataset_name='EasyCV/small_coco_for_test',
            split='train',
            use_streaming=True,
            download_mode=DownloadMode.FORCE_REDOWNLOAD)
        dataset_sample_dict = next(iter(small_coco_for_test))
        print(dataset_sample_dict)
        assert dataset_sample_dict.values()


if __name__ == '__main__':
    unittest.main()
