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
"""Utilities for dealing with writing json strings.

json_utils wraps json.dump and json.dumps so that they can be used to safely
control the precision of floats when writing to json strings or files.
"""
import json
import sys
from json import encoder

import numpy as np

# python 3.5 and newer version does not have json.encoder.FLOAT_REPR
needs_class_hack = sys.version_info >= (3, 5)

encoder.FLOAT_REPR = float.__repr__
INFINITY = float('inf')


class MyEncoder(encoder.JSONEncoder):

    def default(self, o):
        """Implement this method in a subclass such that it returns
    a serializable object for ``o``, or calls the base implementation
    (to raise a ``TypeError``).

    For example, to support arbitrary iterators, you could
    implement default like this::

        def default(self, o):
            try:
                iterable = iter(o)
            except TypeError:
                pass
            else:
                return list(iterable)
            # Let the base class default method raise the TypeError
            return JSONEncoder.default(self, o)

    """
        vtype = type(o)
        if isinstance(o, bytes):
            return str(o, encoding='utf-8')
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif vtype in [np.int16, np.int32, np.int64, np.float32, np.float64]:
            return o.item()
        else:
            return encoder.JSONEncoder.default(self, o)

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
    representation as available.

    For example::

        for chunk in JSONEncoder().iterencode(bigobject):
            mysocket.write(chunk)

    """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encoder.encode_basestring_ascii
        else:
            _encoder = encoder.encode_basestring

        def floatstr(o,
                     allow_nan=self.allow_nan,
                     _repr=encoder.FLOAT_REPR,
                     _inf=INFINITY,
                     _neginf=-INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = 'NaN'
            elif o == _inf:
                text = 'Infinity'
            elif o == _neginf:
                text = '-Infinity'
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    'Out of range float values are not JSON compliant: ' +
                    repr(o))

            return text

        if (_one_shot and encoder.c_make_encoder is not None
                and self.indent is None):
            _iterencode = encoder.c_make_encoder(markers, self.default,
                                                 _encoder, self.indent,
                                                 self.key_separator,
                                                 self.item_separator,
                                                 self.sort_keys, self.skipkeys,
                                                 self.allow_nan)
        else:
            _iterencode = encoder._make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)


def dump(obj, fid, float_digits=-1, **params):
    """Wrapper of json.dump that allows specifying the float precision used.

  Args:
    obj: The object to dump.
    fid: The file id to write to.
    float_digits: The number of digits of precision when writing floats out.
    **params: Additional parameters to pass to json.dumps.
  """
    original_encoder = encoder.FLOAT_REPR
    if float_digits >= 0:
        encoder.FLOAT_REPR = lambda o: format(o, '.%df' % float_digits)
    try:
        if needs_class_hack and 'cls' not in params:
            params['cls'] = MyEncoder
        json.dump(obj, fid, **params)
    finally:
        encoder.FLOAT_REPR = original_encoder


def dumps(obj, float_digits=-1, **params):
    """Wrapper of json.dumps that allows specifying the float precision used.

  Args:
    obj: The object to dump.
    float_digits: The number of digits of precision when writing floats out.
    **params: Additional parameters to pass to json.dumps.

  Returns:
    output: JSON string representation of obj.
  """
    original_encoder = encoder.FLOAT_REPR
    original_c_make_encoder = encoder.c_make_encoder
    if float_digits >= 0:
        encoder.FLOAT_REPR = lambda o: format(o, '.%df' % float_digits)
        encoder.c_make_encoder = None
    try:
        if needs_class_hack and 'cls' not in params:
            params['cls'] = MyEncoder
        output = json.dumps(obj, **params)

    finally:
        encoder.FLOAT_REPR = original_encoder
        encoder.c_make_encoder = original_c_make_encoder

    return output


def compat_dumps(data, float_digits=-1):
    """
  handle json dumps chinese and numpy data
  Args:
    data python data structure
    float_digits: The number of digits of precision when writing floats out.
  Return:
    json str, in python2 , the str is encoded with utf8
      in python3, the str is unicode type(python3 str)
  """
    result_str = dumps(
        data, float_digits=float_digits, cls=MyEncoder, ensure_ascii=False)
    return result_str


def PrettyParams(**params):
    """Returns parameters for use with Dump and Dumps to output pretty json.

  Example usage:
    ```json_str = json_utils.Dumps(obj, **json_utils.PrettyParams())```
    ```json_str = json_utils.Dumps(
                      obj, **json_utils.PrettyParams(allow_nans=False))```

  Args:
    **params: Additional params to pass to json.dump or json.dumps.

  Returns:
    params: Parameters that are compatible with json_utils.Dump and
      json_utils.Dumps.
  """
    params['float_digits'] = 4
    params['sort_keys'] = True
    params['indent'] = 2
    params['separators'] = (',', ': ')
    return params
