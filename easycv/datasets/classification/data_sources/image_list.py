# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import logging
import os

from PIL import Image, ImageFile

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from easycv.file.image import load_image
from easycv.framework.errors import TypeError, ValueError
from easycv.utils.dist_utils import dist_zero_exec
from .utils import split_listfile_byrank


@DATASOURCES.register_module
class ClsSourceImageList(object):
    """ data source for classification
        Args:
            list_file : str / list(str), str means a input image list file path,
                this file contains records as  `image_path label` in list_file
                list(str) means multi image list, each one contains some records as `image_path label`
            root: str / list(str), root path for image_path, each list_file will need a root,
                if len(root) < len(list_file), we will use root[-1] to fill root list.
            delimeter: str, delimeter of each line in the `list_file`
            split_huge_listfile_byrank: Adapt to the situation that the memory cannot fully load a huge amount of data list.
                If split, data list will be split to each rank.
            split_label_balance: if `split_huge_listfile_byrank` is true, whether split with label balance
            cache_path: if `split_huge_listfile_byrank` is true, cache list_file will be saved to cache_path.
    """

    def __init__(self,
                 list_file,
                 root='',
                 delimeter=' ',
                 split_huge_listfile_byrank=False,
                 split_label_balance=False,
                 cache_path='data/',
                 class_list=None):

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # DistributedMPSampler need this attr
        self.has_labels = True
        self.class_list = class_list
        if self.class_list is None:
            logging.warning(
                'It is recommended to specify the ``class_list`` parameter!')
            self.label_dict = {}
        else:
            self.label_dict = dict(
                zip(self.class_list, range(len(self.class_list))))

        if isinstance(list_file, str):
            assert isinstance(root, str), 'list_file is str, root must be str'
            list_file = [list_file]
            root = [root]
        else:
            assert isinstance(list_file, list), \
                'list_file should be str or list(str)'
            root = [root] if isinstance(root, str) else root
            if not isinstance(root, list):
                raise TypeError('root must be str or list(str), but get %s' %
                                type(root))

            if len(root) < len(list_file):
                logging.warning(
                    'len(root) < len(list_file), fill root with root last!')
                root = root + [root[-1]] * (len(list_file) - len(root))

        # TODO: support return list, donot save split file
        # TODO: support loading list_file that have already been split
        if split_huge_listfile_byrank:
            with dist_zero_exec():
                list_file = split_listfile_byrank(
                    list_file=list_file,
                    label_balance=split_label_balance,
                    save_path=cache_path)

        self.fns = []
        self.labels = []
        for l, r in zip(list_file, root):
            fns, labels = self.parse_list_file(l, r, delimeter,
                                               self.label_dict)
            self.fns += fns
            self.labels += labels

    @staticmethod
    def parse_list_file(list_file, root, delimeter, label_dict={}):
        with io.open(list_file, 'r') as f:
            lines = f.readlines()

        fns = []
        labels = []

        for l in lines:
            splits = l.strip().split(delimeter)
            if len(root) > 0:
                fns.append(os.path.join(root, splits[0]))
            else:
                fns.append(splits[0])
            if len(label_dict) == 0:
                # must be int,other with mmcv collect will crash
                label = [int(i) for i in splits[1:]]
            else:
                label = [label_dict[i] for i in splits[1:]]
            labels.append(
                label[0]) if len(label) == 1 else labels.append(label)

        return fns, labels

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        img = load_image(self.fns[idx], mode='RGB')
        if img is None:
            return self[idx + 1]

        img = Image.fromarray(img)
        label = self.labels[idx]

        result_dict = {'img': img, 'gt_labels': label}
        return result_dict


@DATASOURCES.register_module
class ClsSourceItag(ClsSourceImageList):
    """ data source itag for classification
        Args:
            list_file : str / list(str), str means a input image list file path,
                this file contains records as  `image_path label` in list_file
                list(str) means multi image list, each one contains some records as `image_path label`
    """

    def __init__(self, list_file, root='', class_list=None):
        assert root is None or len(
            root) < 1, 'The "root" param is not used and will be removed soon!'
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # DistributedMPSampler need this attr
        self.has_labels = True
        self.class_list = class_list
        if self.class_list is None:
            logging.warning(
                'It is recommended to specify the ``class_list`` parameter!')
            self._auto_collect_labels = True
            self.label_dict = {}
        else:
            self.label_dict = dict(
                zip(self.class_list, range(len(self.class_list))))
            self._auto_collect_labels = False
        self.fns, self.labels, self.label_dict = self.parse_list_file(
            list_file, self.label_dict, self._auto_collect_labels)

    @staticmethod
    def parse_list_file(list_file, label_dict, auto_collect_labels=True):
        with io.open(list_file, 'r') as f:
            rows = f.read().splitlines()

        fns = []
        labels_id = []

        for row_str in rows:
            data_i = json.loads(row_str.strip())
            img_path = data_i['data']['source']
            label_id = []

            priority = 2
            for k in data_i.keys():
                if 'verify' in k:
                    priority = 0
                    break
                elif 'check' in k:
                    priority = 1

            for k, v in data_i.items():
                if 'label' in k:
                    label_id = []
                    result_list = v['results']
                    for j in range(len(result_list)):
                        label = result_list[j]['data']
                        if 'labels' in label:
                            label = label['labels']['单选']
                        if label not in label_dict:
                            if auto_collect_labels:
                                label_dict[label] = len(label_dict)
                            else:
                                raise ValueError(
                                    f'Not find label "{label}" in label dict: {label_dict}'
                                )
                        label_id.append(label_dict[label])
                    if 'verify' in k:
                        break
                    elif 'check' in k and priority == 1:
                        break

            fns.append(img_path)
            labels_id.append(label_id[0]) if len(
                label_id) == 1 else labels_id.append(label_id)

        return fns, labels_id, label_dict
