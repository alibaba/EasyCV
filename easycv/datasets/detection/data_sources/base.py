# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import functools
import logging
from abc import abstractmethod
from multiprocessing import Pool, cpu_count
import os
import wget
import numpy as np
from mmcv.runner.dist_utils import get_dist_info
from tqdm import tqdm

from easycv.file.image import load_image
from easycv.framework.errors import NotImplementedError, ValueError

# The location where downloaded data is stored
DATASET_HOME = os.path.expanduser("~/.cache/easycv/dataset")

'''
    { key : value key: value, key: value, ..... } 
    parameter:
        key  : str
        value: tuple
            explain ：[links, cmd, condition, data_home]
                links: list , collection of data download links
                cmd: str, Data decompression command
                condition: bool, whether to create data_name path, need if True else not need
                data_home: The location where the data is stored after decompression
'''

DATASETS = {

    "small_coco_itag": (
                    ["http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/unittest/data/detection/small_coco_itag/small_coco_itag.tar.gz"],
                    "tar -xzvf ",
                    True,
                    "small_coco_itag"
                ),
    "voc2007": (
                    ["http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"],
                    "tar -xvf ",
                    False,
                    "VOCdevkit/VOC2007"
                ),
    "voc2012": (
                    ["http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"],
                    "tar -xvf ",
                    False,
                    "VOCdevkit/VOC2012/"
                ),
    'coco2017': (
                    [
                        'http://images.cocodataset.org/zips/train2017.zip',
                        'http://images.cocodataset.org/zips/val2017.zip',
                        'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                    ],
                    "unzip -d",
                    True,
                    "COCO2017"
                )
}


def download_file(data_name, dataset_home=DATASET_HOME):
    '''
    data_name: download file of name
    dataset_home: data root path
    '''
    data_name = data_name.lower()
    assert data_name in DATASETS.keys(), f"{data_name} is not down link"
    data_cfg = DATASETS[data_name]
    os.makedirs(dataset_home, exist_ok=True)
    download_finished = list()
    tmp_data = data_cfg[3]
    for link_list in data_cfg[0]:
        filename = wget.filename_from_url(link_list)
        download_finished.append(filename)
        if not os.path.exists(os.path.join(dataset_home, filename)):
            try:
                print(f"{filename} is start download........")
                filename = wget.download(link_list, out=dataset_home)
                print(f"{filename} is download finished\n")
            except:
                print(f"{filename} is download fail")
                exit()

        # The prevention of Ctrol + C
        if not os.path.exists(os.path.join(dataset_home, filename)):
            exit()
    if os.path.exists(os.path.join(dataset_home, tmp_data)):
        return os.path.join(dataset_home, tmp_data)

    for tmp_file in download_finished:
        if data_cfg[2]:
            save_dir = os.path.join(dataset_home, tmp_data)
            os.makedirs(save_dir, exist_ok=True)
            if tmp_file.endswith('zip'):
                cmd = f"{data_cfg[1]} {save_dir} {os.path.join(dataset_home, tmp_file)}"
            else:
                cmd = f"{data_cfg[1]} {os.path.join(dataset_home, tmp_file)} -C {save_dir}"
        else:
            cmd = f"{data_cfg[1]} {os.path.join(dataset_home, tmp_file)} -C {dataset_home}"
        print("begin Unpack.....................")
        os.system(cmd)
        print("Unpack is finished.....................")

    return os.path.join(dataset_home, data_cfg[3])


def _load_image(img_path):
    result = {}
    img = load_image(img_path, mode='BGR')

    result['img'] = img.astype(np.float32)
    result['img_shape'] = img.shape  # h, w, c
    result['ori_img_shape'] = img.shape

    return result


def build_sample(source_item, classes, parse_fn, load_img):
    """Build sample info from source item.
    Args:
        source_item: item of source iterator
        classes: classes list
        parse_fn: parse function to parse source_item, only accepts two params: source_item and classes
        load_img: load image or not, if true, cache all images in memory at init
    """
    result_dict = parse_fn(source_item, classes)

    if load_img:
        result_dict.update(_load_image(result_dict['filename']))

    return result_dict


class DetSourceBase(object):

    def __init__(self,
                 classes=[],
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 parse_fn=None,
                 num_processes=int(cpu_count() / 2),
                 **kwargs):
        """
        Args:
            classes: classes list
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            parse_fn: parse function to parse source iterator, parse_fn should return dict containing:
                gt_bboxes(np.ndarry): Float32 numpy array of shape [num_boxes, 4] and
                        format [ymin, xmin, ymax, xmax] in absolute image coordinates.
                gt_labels(np.ndarry): Integer numpy array of shape [num_boxes]
                    containing 1-indexed detection classes for the boxes.
                filename(str): absolute file path.
            num_processes: number of processes to parse samples
        """
        self.CLASSES = classes
        self.rank, self.world_size = get_dist_info()
        self.cache_at_init = cache_at_init
        self.cache_on_the_fly = cache_on_the_fly
        self.num_processes = num_processes

        if self.cache_at_init and self.cache_on_the_fly:
            raise ValueError(
                'Only one of `cache_on_the_fly` and `cache_at_init` can be True!'
            )
        source_iter = self.get_source_iterator()

        process_fn = functools.partial(
            build_sample,
            parse_fn=parse_fn,
            classes=self.CLASSES,
            load_img=cache_at_init == True,
        )
        self.samples_list = self.build_samples(
            source_iter, process_fn=process_fn)
        self.num_samples = len(self.samples_list)
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

    def __len__(self):
        return len(self.samples_list)

    def get_ann_info(self, idx):
        """
        Get raw annotation info, include bounding boxes, labels and so on.
        `bboxes` format is as [x1, y1, x2, y2] without normalization.
        """
        sample_info = self.samples_list[idx]

        groundtruth_is_crowd = sample_info.get('groundtruth_is_crowd', None)
        if groundtruth_is_crowd is None:
            groundtruth_is_crowd = np.zeros_like(sample_info['gt_labels'])

        annotations = {
            'bboxes': sample_info['gt_bboxes'],
            'labels': sample_info['gt_labels'],
            'groundtruth_is_crowd': groundtruth_is_crowd
        }

        return annotations

    def post_process_fn(self, result_dict):
        if result_dict.get('img_fields', None) is None:
            result_dict['img_fields'] = ['img']
        if result_dict.get('bbox_fields', None) is None:
            result_dict['bbox_fields'] = ['gt_bboxes']

        return result_dict

    def _rand_another(self, idx):
        return (idx + 1) % self.num_samples

    def __getitem__(self, idx):
        result_dict = self.samples_list[idx]
        load_success = True
        try:
            # avoid data cache from taking up too much memory
            if not self.cache_at_init and not self.cache_on_the_fly:
                result_dict = copy.deepcopy(result_dict)

            if not self.cache_at_init and result_dict.get('img', None) is None:
                result_dict.update(_load_image(result_dict['filename']))
                if self.cache_on_the_fly:
                    self.samples_list[idx] = result_dict
            # `post_process_fn` may modify the value of `self.samples_list`,
            # and repeated tries may causing repeated processing operations, which may cause some problems.
            # Use deepcopy to avoid potential problems.
            result_dict = self.post_process_fn(copy.deepcopy(result_dict))
            # load success,reset to 0
            self._retry_count = 0
        except Exception as e:
            logging.error(e)
            load_success = False

        if not load_success:
            logging.warning(
                'Something wrong with current sample %s,Try load next sample...'
                % result_dict.get('filename', ''))
            self._retry_count += 1
            if self._retry_count >= self._max_retry_num:
                raise ValueError('All samples failed to load!')

            result_dict = self[self._rand_another(idx)]

        return result_dict
