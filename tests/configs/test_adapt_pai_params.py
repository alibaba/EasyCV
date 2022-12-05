import os.path as osp
import unittest

from tests.ut_config import CLASS_LIST_TEST

from easycv.utils.config_tools import adapt_pai_params


class AdaptPaiParamsTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_adapt_pai_params_0(self):
        cfg_dict = {}
        class_list_params = ['', 8]
        cfg_dict = adapt_pai_params(
            cfg_dict, class_list_params=class_list_params)

        self.assertEqual(cfg_dict['class_list'],
                         ['0', '1', '2', '3', '4', '5', '6', '7'])

    def test_adapt_pai_params_1(self):
        cfg_dict = {}
        class_list_params = [['person', 'cat', 'dog'], 8]
        cfg_dict = adapt_pai_params(
            cfg_dict, class_list_params=class_list_params)

        self.assertEqual(cfg_dict['class_list'], ['person', 'cat', 'dog'])

    def test_adapt_pai_params_2(self):
        cfg_dict = {}
        class_list_params = [[0, 1, 2], 8]
        cfg_dict = adapt_pai_params(
            cfg_dict, class_list_params=class_list_params)

        self.assertEqual(cfg_dict['class_list'], ['0', '1', '2'])

    def test_adapt_pai_params_3(self):
        cfg_dict = {}
        class_list_params = [
            osp.join(CLASS_LIST_TEST, 'class_list_int_test.txt'), 8
        ]
        cfg_dict = adapt_pai_params(
            cfg_dict, class_list_params=class_list_params)

        self.assertEqual(cfg_dict['class_list'],
                         ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    def test_adapt_pai_params_4(self):
        cfg_dict = {}
        class_list_params = [
            osp.join(CLASS_LIST_TEST, 'class_list_str_test.txt'), 8
        ]
        cfg_dict = adapt_pai_params(
            cfg_dict, class_list_params=class_list_params)

        self.assertEqual(cfg_dict['class_list'], [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ])


if __name__ == '__main__':
    unittest.main()
