import os.path as osp
import unittest

from tests.ut_config import CLASS_LIST_TEST

from easycv.utils.config_tools import update_class_list


class UpdateClassListTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_update_class_list_0(self):
        class_list_params = ['', 8]
        value = update_class_list(class_list_params)

        self.assertEqual(value, ['0', '1', '2', '3', '4', '5', '6', '7'])

    def test_update_class_list_1(self):
        class_list_params = [['person', 'cat', 'dog'], 8]
        value = update_class_list(class_list_params)

        self.assertEqual(value, ['person', 'cat', 'dog'])

    def test_update_class_list_2(self):
        class_list_params = [[0, 1, 2], 8]
        value = update_class_list(class_list_params)

        self.assertEqual(value, ['0', '1', '2'])

    def test_update_class_list_3(self):
        class_list_params = [
            osp.join(CLASS_LIST_TEST, 'class_list_int_test.txt'), 8
        ]
        value = update_class_list(class_list_params)

        self.assertEqual(value,
                         ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    def test_update_class_list_4(self):
        class_list_params = [
            osp.join(CLASS_LIST_TEST, 'class_list_str_test.txt'), 8
        ]
        value = update_class_list(class_list_params)

        self.assertEqual(value, [
            'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck'
        ])


if __name__ == '__main__':
    unittest.main()
