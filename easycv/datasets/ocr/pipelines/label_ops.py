# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os.path as osp

import cv2
import numpy as np
import requests

from easycv.datasets.registry import PIPELINES
from easycv.utils.logger import get_root_logger


@PIPELINES.register_module()
class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False):

        self.max_text_len = max_text_length
        self.BEGIN_STR = 'sos'
        self.END_STR = 'eos'
        self.lower = False

        if character_dict_path is None:
            logger = get_root_logger()
            logger.warning(
                'The character_dict_path is None, model can only recognize number and lower letters'
            )
            self.character_str = '0123456789abcdefghijklmnopqrstuvwxyz'
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            if character_dict_path.startswith('http'):
                r = requests.get(character_dict_path)
                tpath = character_dict_path.split('/')[-1]
                while not osp.exists(tpath):
                    try:
                        with open(tpath, 'wb') as code:
                            code.write(r.content)
                    except:
                        pass
                character_dict_path = tpath
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(' ')
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                # logger = get_logger()
                # logger.warning('{} is not in dict'.format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


@PIPELINES.register_module()
class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        self.BLANK = ['blank']
        super(CTCLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = self.BLANK + dict_character
        return dict_character


@PIPELINES.register_module()
class SARLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        self.BEG_END_STR = '<BOS/EOS>'
        self.UNKNOWN_STR = '<UKN>'
        self.PADDING_STR = '<PAD>'
        super(SARLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.UNKNOWN_STR]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [self.BEG_END_STR]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [self.PADDING_STR]
        self.padding_idx = len(dict_character) - 1

        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data['length'] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]

        padded_text[:len(target)] = target
        data['label'] = np.array(padded_text)
        return data

    def get_ignored_tokens(self):
        return [self.padding_idx]


@PIPELINES.register_module()
class MultiLabelEncode(BaseRecLabelEncode):

    def __init__(self,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        super(MultiLabelEncode,
              self).__init__(max_text_length, character_dict_path,
                             use_space_char)

        self.ctc_encode = CTCLabelEncode(max_text_length, character_dict_path,
                                         use_space_char, **kwargs)
        self.sar_encode = SARLabelEncode(max_text_length, character_dict_path,
                                         use_space_char, **kwargs)

    def __call__(self, data):

        data_ctc = copy.deepcopy(data)
        data_sar = copy.deepcopy(data)
        data_out = dict()
        data_out['img_path'] = data.get('img_path', None)
        data_out['img'] = data['img']
        ctc = self.ctc_encode(data_ctc)
        sar = self.sar_encode(data_sar)
        if ctc is None or sar is None:
            return None
        data_out['label_ctc'] = ctc['label']
        data_out['label_sar'] = sar['label']
        data_out['length'] = ctc['length']
        return data_out
