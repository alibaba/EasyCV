# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.ocr.data_sources.ocr_det_datasource import OCRDetSource
from easycv.datasets.registry import DATASOURCES


@DATASOURCES.register_module()
class OCRClsSource(OCRDetSource):
    """ocr direction classification data source
    """

    def __init__(self,
                 label_file,
                 data_dir='',
                 test_mode=False,
                 delimiter='\t',
                 label_list=['0', '180']):
        """

        Args:
            label_file (str): path of label file
            data_dir (str, optional): folder of imgge data. Defaults to ''.
            test_mode (bool, optional): whether train or test. Defaults to False.
            delimiter (str, optional): delimiter used to separate elements in each row. Defaults to '\t'.
            label_list (list, optional): Identifiable directional Angle. Defaults to ['0', '180'].
        """
        super(OCRClsSource, self).__init__(
            label_file,
            data_dir=data_dir,
            test_mode=test_mode,
            delimiter=delimiter)
        self.label_list = label_list

    def label_encode(self, data):
        label = data['label']
        if label not in self.label_list:
            return None
        label = self.label_list.index(label)
        data['label'] = label
        return data
