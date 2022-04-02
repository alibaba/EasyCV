# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np

from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset


@DATASETS.register_module
class DetDataset(BaseDataset):
    """Dataset for Detection
    """

    def __init__(self, data_source, pipeline, profiling=False, classes=None):
        """
        Args:
            data_source: Data_source config dict
            pipeline: Pipeline config list
            profiling: If set True, will print pipeline time
            classes: A list of class names, used in evaluation for result and groundtruth visualization
        """
        self.classes = classes

        super(DetDataset, self).__init__(
            data_source, pipeline, profiling=profiling)
        self.img_num = self.data_source.get_length()

    def __getitem__(self, idx):
        data_dict = self.data_source.get_sample(idx)
        data_dict = self.pipeline(data_dict)
        return data_dict

    def evaluate(self, results, evaluators, logger=None):
        '''results: a dict of list of Tensors, list length equals to number of test images
        '''
        eval_result = dict()
        annotations = self.data_source.get_labels()
        groundtruth_dict = {}
        groundtruth_dict['groundtruth_boxes'] = [
            labels[:,
                   1:] if len(labels) > 0 else np.array([], dtype=np.float32)
            for labels in annotations
        ]
        groundtruth_dict['groundtruth_classes'] = [
            labels[:, 0] if len(labels) > 0 else np.array([], dtype=np.float32)
            for labels in annotations
        ]
        # bboxes = [label[:, 1:] for label in annotations]
        # scores = [label[:, 0] for label in annotations]
        for evaluator in evaluators:
            eval_result.update(evaluator.evaluate(results, groundtruth_dict))
        # eval_res = {'dummy': 1.0}
        # img = self.data_source.load_ori_img(0)
        # num_box = results['detection_scores'][0].size(0)
        # scores = results['detection_scores'][0].detach().cpu().numpy()
        # bboxes = torch.cat((results['detection_boxes'][0], results['detection_scores'][0].view(num_box, 1)), axis=1).detach().cpu().numpy()
        # labels = results['detection_classes'][0].detach().cpu().numpy().astype(np.int32)
        # # draw bounding boxes
        # score_th = 0.3
        # indices = scores > score_th
        # filter_labels = labels[indices]
        # print([(self.classes[i], score) for i, score in zip(filter_labels, scores)])
        # mmcv.imshow_det_bboxes(
        #     img,
        #     bboxes,
        #     labels,
        #     class_names=self.classes,
        #     score_thr=score_th,
        #     bbox_color='red',
        #     text_color='black',
        #     thickness=1,
        #     font_scale=0.5,
        #     show=False,
        #     wait_time=0,
        #     out_file='test.jpg')

        return eval_result
