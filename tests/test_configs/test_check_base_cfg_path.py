import unittest

from easycv.utils.config_tools import check_base_cfg_path


class CheckPathTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_check_0(self):
        base_cfg_name = 'configs/base.py'
        easycv_root = '/root/easycv'
        father_cfg_name = 'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
        base_cfg_name = check_base_cfg_path(
            base_cfg_name=base_cfg_name,
            father_cfg_name=father_cfg_name,
            easycv_root=easycv_root)

        self.assertEqual(base_cfg_name, '/root/easycv/configs/base.py')

    def test_check_1(self):
        base_cfg_name = 'benchmarks/base.py'
        easycv_root = '/root/easycv'
        father_cfg_name = 'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
        base_cfg_name = check_base_cfg_path(
            base_cfg_name=base_cfg_name,
            father_cfg_name=father_cfg_name,
            easycv_root=easycv_root)

        self.assertEqual(base_cfg_name, '/root/easycv/benchmarks/base.py')

    def test_check_2(self):
        base_cfg_name = '../base.py'
        easycv_root = '/root/easycv'
        father_cfg_name = 'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
        base_cfg_name = check_base_cfg_path(
            base_cfg_name=base_cfg_name,
            father_cfg_name=father_cfg_name,
            easycv_root=easycv_root)

        self.assertEqual(base_cfg_name,
                         'configs/classification/imagenet/base.py')

    def test_check_3(self):
        base_cfg_name = 'common/base.py'
        easycv_root = '/root/easycv'
        father_cfg_name = 'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
        base_cfg_name = check_base_cfg_path(
            base_cfg_name=base_cfg_name,
            father_cfg_name=father_cfg_name,
            easycv_root=easycv_root)

        self.assertEqual(
            base_cfg_name,
            'configs/classification/imagenet/resnet/common/base.py')

    def test_check_4(self):
        base_cfg_name = 'common/base.py'
        easycv_root = '/root/easycv'
        father_cfg_name = 'data/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
        base_cfg_name = check_base_cfg_path(
            base_cfg_name=base_cfg_name,
            father_cfg_name=father_cfg_name,
            easycv_root=easycv_root)

        self.assertEqual(base_cfg_name,
                         'data/classification/imagenet/resnet/common/base.py')

    def test_check_5(self):
        base_cfg_name = '../base.py'
        easycv_root = '/root/easycv'
        father_cfg_name = 'data/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
        base_cfg_name = check_base_cfg_path(
            base_cfg_name=base_cfg_name,
            father_cfg_name=father_cfg_name,
            easycv_root=easycv_root)

        self.assertEqual(base_cfg_name, 'data/classification/imagenet/base.py')


if __name__ == '__main__':
    unittest.main()
