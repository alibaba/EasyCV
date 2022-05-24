# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import functools
import logging
from abc import abstractmethod
from multiprocessing import Pool, cpu_count

import cv2
import mmcv
import numpy as np
from tqdm import tqdm

from easycv.datasets.registry import DATASOURCES
from easycv.file.image import load_image as _load_img


def load_image(img_path):
    img = _load_img(img_path, mode='RGB')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result = {
        'img': img.astype(np.float32),
        'img_shape': img.shape,  # h, w, c
        'ori_shape': img.shape,
    }
    return result


def load_seg_map(seg_path, reduce_zero_label):
    gt_semantic_seg = _load_img(seg_path, mode='RGB')
    # reduce zero_label
    if reduce_zero_label:
        # avoid using underflow conversion
        gt_semantic_seg[gt_semantic_seg == 0] = 255
        gt_semantic_seg = gt_semantic_seg - 1
        gt_semantic_seg[gt_semantic_seg == 254] = 255

    return {'gt_semantic_seg': gt_semantic_seg}


def build_sample(source_item, classes, parse_fn, load_img, reduce_zero_label):
    """Build sample info from source item.
    Args:
        source_item: item of source iterator
        classes: classes list
        parse_fn: parse function to parse source_item, only accepts two params: source_item and classes
        load_img: load image or not, if true, cache all images in memory at init
    """
    result_dict = parse_fn(source_item, classes)

    if load_img:
        result_dict.update(load_image(result_dict['filename']))
        result_dict.update(
            load_seg_map(result_dict['seg_filename'], reduce_zero_label))

    return result_dict


@DATASOURCES.register_module
class SegSourceBase(object):
    """Data source for semantic segmentation.
        classes (str | list): classes list or file
        reduce_zero_label (bool): whether to mark label zero as ignored
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            palette of segmentation map, if none, random palette will be generated
        num_processes: number of processes to parse samples
        cache_at_init (bool): if set True, will cache in memory in __init__ for faster training
        cache_on_the_fly (bool): if set True, will cache in memroy during training
    """
    CLASSES = None
    PALETTE = None

    def __init__(self,
                 classes=None,
                 reduce_zero_label=False,
                 palette=None,
                 parse_fn=None,
                 num_processes=int(cpu_count() / 2),
                 cache_at_init=False,
                 cache_on_the_fly=False):

        if classes is not None:
            self.CLASSES = classes
        if palette is not None:
            self.PALETTE = palette

        self.reduce_zero_label = reduce_zero_label
        self.cache_at_init = cache_at_init
        self.cache_on_the_fly = cache_on_the_fly
        self.num_processes = num_processes

        if self.cache_at_init and self.cache_on_the_fly:
            raise ValueError(
                'Only one of `cache_on_the_fly` and `cache_at_init` can be True!'
            )
        assert isinstance(self.CLASSES, (str, tuple, list))
        if isinstance(self.CLASSES, str):
            self.CLASSES = mmcv.list_from_file(classes)
        if self.PALETTE is None:
            self.PALETTE = self.get_random_palette()

        source_iter = self.get_source_iterator()

        process_fn = functools.partial(
            build_sample,
            parse_fn=parse_fn,
            classes=self.CLASSES,
            load_img=cache_at_init == True,
            reduce_zero_label=self.reduce_zero_label)
        self.samples_list = self.build_samples(
            source_iter, process_fn=process_fn)
        self.num_samples = self.get_length()
        # An error will be raised if failed to load _max_retry_num times in a row
        self._max_retry_num = self.num_samples
        self._retry_count = 0

    @abstractmethod
    def get_source_iterator():
        """Return data list iterator, source iterator will be passed to parse_fn,
        and parse_fn will receive params of item of source iter and classes for parsing.
        What does parse_fn need, what does source iterator returns.
        """
        raise NotImplementedError

    def build_samples(self, iterable, process_fn):
        samples_list = []
        with Pool(processes=self.num_processes) as p:
            with tqdm(total=len(iterable), desc='Scanning images') as pbar:
                for _, result_dict in enumerate(
                        p.imap_unordered(process_fn, iterable)):
                    if result_dict:
                        samples_list.append(result_dict)
                    pbar.update()

        return samples_list

    def get_sample(self, idx):
        result_dict = self.samples_list[idx]
        load_success = True
        try:
            if not self.cache_at_init:
                if result_dict.get('img', None) is None:
                    result_dict.update(load_image(result_dict['filename']))
                if result_dict.get('gt_semantic_seg', None) is None:
                    result_dict.update(
                        load_seg_map(
                            result_dict['seg_filename'],
                            reduce_zero_label=self.reduce_zero_label))
                if self.cache_on_the_fly:
                    self.samples_list[idx] = result_dict
            result_dict = self.post_process_fn(copy.deepcopy(result_dict))
            self._retry_count = 0
        except Exception as e:
            logging.warning(e)
            load_success = False

        if not load_success:
            logging.warning(
                'Something wrong with current sample %s,Try load next sample...'
                % result_dict.get('filename', ''))
            self._retry_count += 1
            if self._retry_count >= self._max_retry_num:
                raise ValueError('All samples failed to load!')

            result_dict = self.get_sample((idx + 1) % self.num_samples)

        return result_dict

    def post_process_fn(self, result_dict):
        if result_dict.get('img_fields', None) is None:
            result_dict['img_fields'] = ['img']
        if result_dict.get('seg_fields', None) is None:
            result_dict['seg_fields'] = ['gt_semantic_seg']

        return result_dict

    def get_random_palette(self):
        # Get random state before set seed, and restore
        # random state later.
        # It will prevent loss of randomness, as the palette
        # may be different in each iteration if not specified.
        # See: https://github.com/open-mmlab/mmdetection/issues/5844
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(0, 255, size=(len(self.CLASSES), 3))
        np.random.set_state(state)

        return palette

    def __len__(self):
        return self.get_length()

    def get_length(self):
        return len(self.samples_list)
