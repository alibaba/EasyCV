# Copyright (c) Alibaba, Inc. and its affiliates.
import math

from mmcv.runner import get_dist_info

from easycv.datasets.builder import build_datasource
from easycv.datasets.loader.sampler import DistributedSampler
from easycv.datasets.registry import DATASETS, PIPELINES
from easycv.datasets.shared.pipelines.transforms import Compose
from easycv.utils import dist_utils
from easycv.utils.registry import build_from_cfg


class DaliLoaderWrapper(object):

    def __init__(self, dali_loader, batch_size, return_list):
        self.dali_loader = dali_loader
        self.batch_size = batch_size
        self.return_list = return_list
        self.loader = None
        rank, world_size = get_dist_info()
        self.rank = rank
        self.sampler = DistributedSampler(
            [1] * 100, world_size, rank, shuffle=True, replace=False)

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

        img = [
            data[0][self.return_list[i]]
            for i in range(len(self.return_list) - 1)
        ]

        return dict(img=img)


def _load_DaliTFRecordMultiViewPipe():
    """Avoid import nvidia errors when not use.
    """

    import nvidia.dali.ops as ops
    from nvidia.dali.pipeline import Pipeline
    from easycv.datasets.utils.tfrecord_util import get_imagenet_dali_tfrecord_feature

    class DaliTFRecordMultiViewPipe(Pipeline):

        def __init__(self,
                     data_source,
                     transforms_list,
                     batch_size,
                     distributed,
                     random_shuffle=True,
                     workers_per_gpu=2,
                     device='gpu'):

            self.device = device

            if distributed:
                self.rank, self.world_size = get_dist_info()
                self.local_rank = dist_utils.local_rank()
                self.local_size = dist_utils.get_num_gpu_per_node()
            else:
                self.rank, self.local_rank, self.world_size = 0, 0, 1

            super(DaliTFRecordMultiViewPipe, self).__init__(
                batch_size,
                workers_per_gpu,
                self.local_rank,
                seed=12 + self.rank)

            imagenet_feature = get_imagenet_dali_tfrecord_feature()
            self.input = ops.TFRecordReader(
                path=data_source.data_list,
                index_path=data_source.index_list,
                shard_id=self.rank,
                num_shards=self.world_size,
                random_shuffle=random_shuffle,
                features=imagenet_feature)

            self.transforms_list = transforms_list

        def define_graph(self):
            inputs = self.input(name='Reader')
            jpegs, labels = inputs['image/encoded'], inputs[
                'image/class/label']

            output_list = []
            for t_i in self.transforms_list:
                output_list.append(t_i(jpegs))

            if self.device == 'gpu':
                labels = labels.gpu()
            output_list.append(labels)

            return output_list

    return DaliTFRecordMultiViewPipe


@DATASETS.register_module
class DaliTFRecordMultiViewDataset(object):
    """Adapt to dali, the dataset outputs multiple views of an image.
    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.
    Args:
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines.
    """

    def __init__(self,
                 data_source,
                 num_views,
                 pipelines,
                 distributed,
                 batch_size,
                 random_shuffle=True,
                 workers_per_gpu=2):

        assert len(num_views) == len(pipelines)
        self.batch_size = batch_size

        if distributed:
            self.rank, self.world_size = get_dist_info()
        else:
            self.rank, self.world_size = 0, 1

        data_source = build_datasource(data_source)

        transforms = []
        for pipeline in pipelines:
            transform = Compose(
                [build_from_cfg(p, PIPELINES) for p in pipeline])
            transforms.append(transform)

        self.transforms_list = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            self.transforms_list.extend([transforms[i]] * num_views[i])

        DaliTFRecordMultiViewPipe = _load_DaliTFRecordMultiViewPipe()
        self.pipeline = DaliTFRecordMultiViewPipe(
            data_source=data_source,
            transforms_list=self.transforms_list,
            batch_size=batch_size,
            distributed=distributed,
            random_shuffle=random_shuffle,
            workers_per_gpu=workers_per_gpu)

    def get_dataloader(self):
        from nvidia.dali.plugin.pytorch import DALIGenericIterator

        batch_size = self.batch_size

        self.pipeline.build()
        data_size = int(self.pipeline.epoch_size('Reader') / self.world_size)
        return_list = [
            'image_%s' % i for i in range(len(self.transforms_list))
        ] + ['label']

        data_loader = DALIGenericIterator([self.pipeline],
                                          return_list,
                                          data_size,
                                          fill_last_batch=True)
        data_loader = DaliLoaderWrapper(data_loader, batch_size, return_list)
        return data_loader
