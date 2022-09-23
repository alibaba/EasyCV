# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import traceback

import cv2
import lmdb
import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file.image import load_image


@DATASOURCES.register_module()
class OCRRecSource(object):
    """ocr rec data source
    """

    def __init__(self,
                 label_file,
                 data_dir='',
                 ext_data_num=0,
                 test_mode=False,
                 delimiter='\t'):
        """

        Args:
            label_file (str): path of label file
            data_dir (str, optional): folder of imgge data. Defaults to ''.
            ext_data_num (int): number of additional data used for augmentation. Defaults to 0.
            test_mode (bool, optional): whether train or test. Defaults to False.
            delimiter (str, optional): delimiter used to separate elements in each row. Defaults to '\t'.
        """
        self.data_dir = data_dir
        self.delimiter = delimiter
        self.test_mode = test_mode
        self.ext_data_num = ext_data_num
        self.data_lines = self.get_image_info_list(label_file)

    def get_image_info_list(self, label_file):
        data_lines = []
        with open(label_file, 'rb') as f:
            lines = f.readlines()
            data_lines.extend(lines)
        return data_lines

    def __getitem__(self, idx, get_ext=True):
        data_line = self.data_lines[idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip('\n').split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            outs = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception('{} does not exist!'.format(img_path))
            img = load_image(img_path, mode='BGR')
            outs['img'] = img.astype(np.float32)
            outs['ori_img_shape'] = img.shape
            if get_ext:
                outs['ext_data'] = self.get_ext_data()
            return outs
        except:
            logging.error(
                'When parsing line {}, error happened with msg: {}'.format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            rnd_idx = np.random.randint(self.__len__(
            )) if not self.test_mode else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.data_lines)

    def get_ext_data(self):
        ext_data = []

        while len(ext_data) < self.ext_data_num:
            data = self.__getitem__(
                np.random.randint(self.__len__()), get_ext=False)
            ext_data.append(data)
        return ext_data


@DATASOURCES.register_module(force=True)
class OCRReclmdbSource(object):
    """ocr rec lmdb data source specific for DTRB dataset
    """

    def __init__(self, data_dir='', ext_data_num=0, test_mode=False):
        self.test_mode = test_mode
        self.ext_data_num = ext_data_num
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        logging.info('Initialize indexs of datasets:%s' % data_dir)
        self.data_idx_order_list = self.dataset_traversal()

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {
                    'dirpath': dirpath,
                    'env': env,
                    'txn': txn,
                    'num_samples': num_samples
                }
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx, get_ext=True):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        if sample_info is None:
            rnd_idx = np.random.randint(self.__len__(
            )) if not self.test_mode else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        img, label = sample_info
        img = cv2.imdecode(np.frombuffer(img, dtype='uint8'), 1)
        outs = {'img_path': '', 'label': label}
        outs['img'] = img.astype(np.float32)
        outs['ori_img_shape'] = img.shape
        if get_ext:
            outs['ext_data'] = self.get_ext_data()
        return outs

    def get_ext_data(self):
        ext_data = []

        while len(ext_data) < self.ext_data_num:
            data = self.__getitem__(
                np.random.randint(self.__len__()), get_ext=False)
            ext_data.append(data)
        return ext_data

    def __len__(self):
        return self.data_idx_order_list.shape[0]
