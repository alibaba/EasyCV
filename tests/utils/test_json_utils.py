#! -*- coding: utf8 -*-

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for google3.image.understanding.utils.json_utils."""
import json
import os
import tempfile
import unittest

from easycv.file import io
from easycv.utils import json_utils


class JsonUtilsTest(unittest.TestCase):

    def testDumpReasonablePrecision(self):
        output_path = os.path.join(tempfile.gettempdir(), 'test.json')
        with io.open(output_path, 'w') as f:
            json_utils.dump(1.0, f, float_digits=2)
        with io.open(output_path, 'r') as f:
            self.assertEqual(f.read(), '1.00')

    def testDumpPassExtraParams(self):
        output_path = os.path.join(tempfile.gettempdir(), 'test.json')
        with io.open(output_path, 'w') as f:
            json_utils.dump([1.0], f, float_digits=2, indent=3)
        with io.open(output_path, 'r') as f:
            self.assertEqual(f.read(), '[\n   1.00\n]')

    def testDumpZeroPrecision(self):
        output_path = os.path.join(tempfile.gettempdir(), 'test.json')
        with io.open(output_path, 'w') as f:
            json_utils.dump(1.0, f, float_digits=0, indent=3)
        with io.open(output_path, 'r') as f:
            self.assertEqual(f.read(), '1')

    def testDumpUnspecifiedPrecision(self):
        output_path = os.path.join(tempfile.gettempdir(), 'test.json')
        with io.open(output_path, 'w') as f:
            json_utils.dump(1.012345, f)
        with io.open(output_path, 'r') as f:
            self.assertEqual(f.read(), '1.012345')

    def testDumpsReasonablePrecision(self):
        s = json_utils.dumps(1.0, float_digits=2)
        self.assertEqual(s, '1.00')

    def testDumpsPassExtraParams(self):
        s = json_utils.dumps([1.0], float_digits=2, indent=3)
        self.assertEqual(s, '[\n   1.00\n]')

    def testDumpsZeroPrecision(self):
        s = json_utils.dumps(1.0, float_digits=0)
        self.assertEqual(s, '1')

    def testDumpsUnspecifiedPrecision(self):
        s = json_utils.dumps(1.012345)
        self.assertEqual(s, '1.012345')

    def testPrettyParams(self):
        s = json_utils.dumps({
            'v': 1.012345,
            'n': 2
        }, **json_utils.PrettyParams())
        self.assertEqual(s, '{\n  "n": 2,\n  "v": 1.0123\n}')

    def testPrettyParamsExtraParamsInside(self):
        s = json_utils.dumps({
            'v': 1.012345,
            'n': float('nan')
        }, **json_utils.PrettyParams(allow_nan=True))
        self.assertEqual(s, '{\n  "n": NaN,\n  "v": 1.0123\n}')

        with self.assertRaises(ValueError):
            s = json_utils.dumps({
                'v': 1.012345,
                'n': float('nan')
            }, **json_utils.PrettyParams(allow_nan=False))

    def testPrettyParamsExtraParamsOutside(self):
        s = json_utils.dumps({
            'v': 1.012345,
            'n': float('nan')
        },
                             allow_nan=True,
                             **json_utils.PrettyParams())
        self.assertEqual(s, '{\n  "n": NaN,\n  "v": 1.0123\n}')

        with self.assertRaises(ValueError):
            s = json_utils.dumps({
                'v': 1.012345,
                'n': float('nan')
            },
                                 allow_nan=False,
                                 **json_utils.PrettyParams())

    def testDumpsNumpy(self):
        import numpy as np
        data = [np.array([1, 2]) for i in range(5)]
        s = json.dumps(data, cls=json_utils.MyEncoder)
        new_data = json.loads(s)
        self.assertTrue(isinstance(new_data, list))
        self.assertTrue(isinstance(new_data[0], list))

        data = {
            'detection_class_names': np.array(['cls1', 'cl2']),
            'detection_boxes': np.array([[1, 1, 0, 0], [1, 0, 1, 1]]),
            'text': np.array(['中文1', '中文2'])
        }
        s = json_utils.compat_dumps(data)
        self.assertTrue(isinstance(s, str))
        print(s)

    def testCompatDumps(self):
        import numpy as np
        a = np.array([0.123344, 0.12333])
        s = json_utils.compat_dumps(a, float_digits=3)
        self.assertTrue(s, '[0.123, 0.123]')
        print(s)


if __name__ == '__main__':
    unittest.main()
