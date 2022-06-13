# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch
from mmcv.runner import get_dist_info

from easycv.datasets.builder import build_datasource
from easycv.datasets.loader.sampler import DistributedSampler
from easycv.datasets.registry import DATASETS, PIPELINES
from easycv.datasets.shared.pipelines.transforms import Compose
from easycv.utils import dist_utils
from easycv.utils.registry import build_from_cfg


class DaliLoaderWrapper(object):

    def __init__(self, dali_loader, batch_size, label_offset=0):
        self.dali_loader = dali_loader
        self.batch_size = batch_size
        self.loader = None
        rank, world_size = get_dist_info()
        self.rank = rank
        self.sampler = DistributedSampler(
            [1] * 100, world_size, rank, shuffle=True, replace=False)
        self.data_length = self.dali_loader._size * world_size
        self.label_offset = label_offset

    def __len__(self):
        return math.ceil(self.dali_loader._size / self.batch_size)

    def __iter__(self):
        self.loader = iter(self.dali_loader)
        return self

    def __next__(self):
        try:
            data = next(self.loader)
        except StopIteration:
            self.dali_loader.reset()
            raise StopIteration

        # tfrecord label is in [1, 1000], which should be [0, 999]
        # return {'img':data[0]["data"], 'gt_label': (data[0]["label"]- torch.ones_like(data[0]["label"])).squeeze().cuda().long()}
        if self.label_offset == 0:
            return {
                'img': data[0]['data'],
                'gt_labels': (data[0]['label']).squeeze().cuda().long()
            }
        else:
            return {
                'img':
                data[0]['data'],
                'gt_labels':
                (data[0]['label'] - self.label_offset *
                 torch.ones_like(data[0]['label'])).squeeze().cuda().long()
            }

    def evaluate(self, results, evaluators, logger=None):
        '''evaluate classification task

        Args:
            results: a dict of list of tensor, including prediction and groundtruth
                info, where prediction tensor is NxC，and the same with groundtruth labels.

            evaluators: a list of evaluator

        Return:
            eval_result: a dict of float, different metric values
        '''
        assert len(
            evaluators
        ) == 1, 'classification evaluation only support one evaluator'
        gt_labels = results.pop('gt_labels')
        eval_res = evaluators[0].evaluate(results, gt_labels)

        return eval_res


def _load_ImageNetTFRecordPipe():
    """Avoid import nvidia errors when not use.
    """

    import nvidia.dali.ops as ops
    from nvidia.dali.pipeline import Pipeline
    from easycv.datasets.utils.tfrecord_util import get_imagenet_dali_tfrecord_feature

    imagenet_feature = get_imagenet_dali_tfrecord_feature()

    class ImageNetTFRecordPipe(Pipeline):

        def __init__(self,
                     data_source,
                     transforms,
                     batch_size,
                     distributed,
                     random_shuffle=True,
                     workers_per_gpu=2,
                     device='gpu'):

            self.device = device

            if distributed:
                self.rank, self.world_size = get_dist_info()
                self.local_rank = dist_utils.local_rank()
            else:
                self.rank, self.local_rank, self.world_size = 0, 0, 1

            super(ImageNetTFRecordPipe, self).__init__(
                batch_size,
                workers_per_gpu,
                self.local_rank,
                seed=12 + self.rank)

            self.input = ops.TFRecordReader(
                path=data_source.data_list,
                index_path=data_source.index_list,
                shard_id=self.rank,
                num_shards=self.world_size,
                random_shuffle=random_shuffle,
                features=imagenet_feature)

            self.transforms = transforms

        def define_graph(self):
            inputs = self.input(name='Reader')
            jpegs, labels = inputs['image/encoded'], inputs[
                'image/class/label']

            images = self.transforms(jpegs)

            if self.device == 'gpu':
                labels = labels.gpu()

            return [images, labels]

    return ImageNetTFRecordPipe


@DATASETS.register_module
class DaliImageNetTFRecordDataSet(object):

    def __init__(self,
                 data_source,
                 pipeline,
                 distributed,
                 batch_size,
                 label_offset=0,
                 random_shuffle=True,
                 workers_per_gpu=2):

        if distributed:
            self.rank, self.world_size = get_dist_info()
        else:
            self.rank, self.world_size = 0, 1
        self.batch_size = batch_size

        data_source = build_datasource(data_source)
        transforms = [build_from_cfg(p, PIPELINES) for p in pipeline]
        transforms = Compose(transforms)

        ImageNetTFRecordPipe = _load_ImageNetTFRecordPipe()
        self.dali_pipe = ImageNetTFRecordPipe(
            data_source=data_source,
            transforms=transforms,
            batch_size=batch_size,
            distributed=distributed,
            random_shuffle=random_shuffle,
            workers_per_gpu=workers_per_gpu)

        self.label_offset = label_offset

    def get_dataloader(self):
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator

        self.dali_pipe.build()
        data_size = int(self.dali_pipe.epoch_size('Reader') / self.world_size)
        data_loader = DALIClassificationIterator([self.dali_pipe],
                                                 size=data_size)
        data_loader = DaliLoaderWrapper(data_loader, self.batch_size,
                                        self.label_offset)

        return data_loader
