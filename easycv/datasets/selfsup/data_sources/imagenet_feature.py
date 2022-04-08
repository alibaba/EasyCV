# Copyright (c) Alibaba, Inc. and its affiliates.
from glob import glob

import numpy as np
from tqdm import tqdm

from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class SSLSourceImageNetFeature(object):

    def __init__(self,
                 root_path,
                 training=True,
                 data_keyword='feat1',
                 label_keyword='label',
                 dynamic_load=True):
        self.training = training
        self.dynamic_load = dynamic_load

        mode = 'train' if training else 'val'

        # train feature save in block, root_path/train_idx(xxx)_keyword.npy,
        if mode == 'train':
            self.embs_list = sorted(
                glob('%s/%s*%s*' % (root_path, mode, data_keyword)),
                key=lambda a: int(a.split('/')[-1].split('_')[1][3:]))
            self.labels_list = sorted(
                glob('%s/%s*%s*' % (root_path, mode, label_keyword)),
                key=lambda a: int(a.split('/')[-1].split('_')[1][3:]))
        else:
            self.embs_list = glob('%s/%s*%s*' %
                                  (root_path, mode, data_keyword))
            self.labels_list = glob('%s/%s*%s*' %
                                    (root_path, mode, label_keyword))

        # for imagenet we decide to load all feature into memory, 2048 should allocate > 8G
        assert len(self.embs_list) == len(self.labels_list)
        assert len(self.embs_list) > 0

        # load to memory is too slow
        # TODO: multiprocess loading to accelerate
        if not dynamic_load:
            self.embs = np.load(self.embs_list[0])
            self.labels = np.load(self.labels_list[0])
            pt = tqdm(zip(self.embs_list[1:], self.labels_list[1:]))
            # for embs_path, label_path in zip(embs_list[1:], labels_list[1:]):
            for embs_path, label_path in pt:
                # print(embs_path, label_path)
                cur_embs = np.load(embs_path)
                cur_label = np.load(label_path)
                self.embs = np.concatenate((self.embs, cur_embs))
                self.labels = np.concatenate((self.labels, cur_label))

        # do a little cache version
        else:
            if np.load(self.embs_list[0]).shape[0] == 0:
                self.embs_list = self.embs_list[1:]
                self.labels_list = self.labels_list[1:]

            # count total samples by labels
            self.labels = np.load(self.labels_list[0])
            pt = tqdm(self.labels_list[1:])
            for label_path in pt:
                cur_label = np.load(label_path)
                self.labels = np.concatenate((self.labels, cur_label))

            self.embs_cache_dict = {}
            self.labels_cache_dict = {}
            self.feature_per_block = np.load(self.embs_list[0]).shape[0]

    def get_sample(self, idx):
        if not self.dynamic_load:
            results = {'img': self.embs[idx], 'gt_labels': self.labels[idx]}
            return results

        block_idx = int(idx / self.feature_per_block)
        if block_idx not in self.embs_cache_dict:
            self.embs_cache_dict[block_idx] = np.load(
                self.embs_list[block_idx])
            self.labels_cache_dict[block_idx] = np.load(
                self.labels_list[block_idx])

        feature = self.embs_cache_dict[block_idx][idx % self.feature_per_block]
        label = int(self.labels_cache_dict[block_idx][idx %
                                                      self.feature_per_block])

        results = {'img': feature, 'gt_labels': label}
        return results

    def get_length(self):
        return self.labels.shape[0]
