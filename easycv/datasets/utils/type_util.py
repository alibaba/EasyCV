# Copyright (c) Alibaba, Inc. and its affiliates.


def is_dali_dataset_type(type_name):
    dali_dataset_types = [
        'DaliImageNetTFRecordDataSet', 'DaliTFRecordMultiViewDataset'
    ]
    return type_name in dali_dataset_types
