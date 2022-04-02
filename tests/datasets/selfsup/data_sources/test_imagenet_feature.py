# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

from tests.ut_config import SSL_SMALL_IMAGENET_FEATURE

from easycv.datasets.selfsup.data_sources.imagenet_feature import \
    SSLSourceImageNetFeature


class SSLSourceImageNetFeatureTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_imagenet_feature_dynamic_load(self):

        data_source = SSLSourceImageNetFeature(
            root_path=SSL_SMALL_IMAGENET_FEATURE)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            feat, label = data_source.get_sample(idx)
            self.assertEqual(feat.shape, (2048, ))
            self.assertIn(label, list(range(1000)))

        length = data_source.get_length()
        self.assertEqual(length, 3215)

    def test_imagenet_feature(self):

        data_source = SSLSourceImageNetFeature(
            root_path=SSL_SMALL_IMAGENET_FEATURE, dynamic_load=False)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            feat, label = data_source.get_sample(idx)
            self.assertEqual(feat.shape, (2048, ))
            self.assertIn(label, list(range(1000)))

        length = data_source.get_length()
        self.assertEqual(length, 3215)


if __name__ == '__main__':
    unittest.main()
