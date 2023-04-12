# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest


class ErrorsTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_errors(self):
        from easycv.framework import errors

        def dummy_op():
            pass

        with self.assertRaises(errors.ValueError) as cm:
            raise errors.ValueError(
                'value error', details='provide correct value', op=dummy_op)
        value_exception = cm.exception
        self.assertEqual(value_exception.error_code, hex(errors.INVALID_VALUE))
        self.assertEqual(value_exception.op, dummy_op)
        self.assertEqual(value_exception.details, 'provide correct value')
        self.assertEqual(value_exception.message, 'value error')

        with self.assertRaises(errors.NotImplementedError) as cm:
            raise errors.NotImplementedError()
        value_exception = cm.exception
        self.assertEqual(value_exception.error_code, hex(errors.UNIMPLEMENTED))
        self.assertEqual(value_exception.op, None)
        self.assertEqual(value_exception.details, None)
        self.assertEqual(value_exception.message, '')

        with self.assertRaises(errors.FileNotFoundError) as cm:
            raise errors.FileNotFoundError
        value_exception = cm.exception
        self.assertEqual(value_exception.error_code,
                         hex(errors.FILE_NOT_FOUND))
        self.assertEqual(value_exception.op, None)
        self.assertEqual(value_exception.details, None)
        self.assertEqual(value_exception.message, '')

        with self.assertRaises(errors.TimeoutError) as cm:
            raise errors.TimeoutError('time out')
        value_exception = cm.exception
        self.assertEqual(value_exception.error_code, hex(errors.TIMEOUT))
        self.assertEqual(value_exception.op, None)
        self.assertEqual(value_exception.details, None)
        self.assertEqual(value_exception.message, 'time out')


if __name__ == '__main__':
    unittest.main()
