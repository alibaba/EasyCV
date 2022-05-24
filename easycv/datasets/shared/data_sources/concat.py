# Copyright (c) Alibaba, Inc. and its affiliates.
import bisect

from easycv.datasets.builder import build_datasource
from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module
class SourceConcat(object):
    """Concat multi data source config.
    """

    def __init__(self, data_source_list):
        assert isinstance(data_source_list, (list, tuple)), \
            'data_source_list must be a config list'
        assert len(data_source_list) > 0, \
            'data_source_list should not be an empty list'
        self.source_type = [
            source_cfg['type'] for source_cfg in data_source_list
        ]
        self.data_sources = [
            build_datasource(data_source_cfg)
            for data_source_cfg in data_source_list
        ]
        self.cumsum_length_list = self.cumsum_length()

    def get_length(self):
        return self.cumsum_length_list[-1]

    def __len__(self):
        return self.get_length()

    def get_sample(self, idx):
        dataset_idx = bisect.bisect_right(self.cumsum_length_list, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumsum_length_list[dataset_idx - 1]

        return self.data_sources[dataset_idx].get_sample(sample_idx)

    def cumsum_length(self):
        len_cumsum_list, idx = [], 0
        for ds in self.data_sources:
            ds_l = ds.get_length()
            len_cumsum_list.append(ds_l + idx)
            idx += ds_l
        return len_cumsum_list
