# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np

from easycv.core.visualization.image import imshow_bboxes
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset
from easycv.file.image import load_image


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
        self.CLASSES = classes

        super(DetDataset, self).__init__(
            data_source, pipeline, profiling=profiling)
        self.num_samples = self.data_source.get_length()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        count = 0
        while True:
            if count > 10:
                raise RuntimeError('Loops timeout')
            data_dict = self.data_source.get_sample(idx)
            data_dict = self.pipeline(data_dict)
            if data_dict is None:
                count += 1
                idx = self.data_source._rand_another(idx)
                continue
            return data_dict

    def evaluate(self, results, evaluators=None, logger=None):
        '''Evaluates the detection boxes.
            Args:
                results: A dictionary containing
                    detection_boxes: List of length number of test images.
                        Float32 numpy array of shape [num_boxes, 4] and
                        format [ymin, xmin, ymax, xmax] in absolute image coordinates.
                    detection_scores: List of length number of test images,
                        detection scores for the boxes, float32 numpy array of shape [num_boxes].
                    detection_classes: List of length number of test images,
                        integer numpy array of shape [num_boxes]
                        containing 1-indexed detection classes for the boxes.
                    img_metas: List of length number of test images,
                        dict of image meta info, containing filename, img_shape,
                        origin_img_shape, scale_factor and so on.
                evaluators: evaluators to calculate metric with results and groundtruth_dict
        '''

        eval_result = dict()

        groundtruth_dict = {}
        groundtruth_dict['groundtruth_boxes'] = [
            self.data_source.get_ann_info(idx)['bboxes']
            for idx in range(len(results['img_metas']))
        ]
        groundtruth_dict['groundtruth_classes'] = [
            self.data_source.get_ann_info(idx)['labels']
            for idx in range(len(results['img_metas']))
        ]
        groundtruth_dict['groundtruth_is_crowd'] = [
            self.data_source.get_ann_info(idx)['groundtruth_is_crowd']
            for idx in range(len(results['img_metas']))
        ]
        groundtruth_dict['groundtruth_instance_masks'] = [
            self.data_source.get_ann_info(idx).get('masks', None)
            for idx in range(len(results['img_metas']))
        ]

        for evaluator in evaluators:
            eval_result.update(evaluator.evaluate(results, groundtruth_dict))

        return eval_result

    def visualize(self, results, vis_num=10, score_thr=0.3, **kwargs):
        """Visulaize the model output on validation data.
        Args:
            results: A dictionary containing
                detection_boxes: List of length number of test images.
                    Float32 numpy array of shape [num_boxes, 4] and
                    format [ymin, xmin, ymax, xmax] in absolute image coordinates.
                detection_scores: List of length number of test images,
                    detection scores for the boxes, float32 numpy array of shape [num_boxes].
                detection_classes: List of length number of test images,
                    integer numpy array of shape [num_boxes]
                    containing 1-indexed detection classes for the boxes.
                img_metas: List of length number of test images,
                    dict of image meta info, containing filename, img_shape,
                    origin_img_shape, scale_factor and so on.
            vis_num: number of images visualized
            score_thr: The threshold to filter box,
                boxes with scores greater than score_thr will be kept.
        Returns: A dictionary containing
            images: Visulaized images.
            img_metas: List of length number of test images,
                    dict of image meta info, containing filename, img_shape,
                    origin_img_shape, scale_factor and so on.
        """
        class_names = None
        if hasattr(self.data_source, 'CLASSES'):
            class_names = self.data_source.CLASSES
        elif hasattr(self.data_source, 'classes'):
            class_names = self.data_source.classes

        # If class_names is not None, class_id will be converted to class_name for visualization,
        # otherwise the class_id will be displayed.
        # And don't try to modify the value in results, it may cause some bugs or even precision problems,
        # because `self.evaluate` will also use the results, refer to: https://github.com/alibaba/EasyCV/pull/67
        if class_names is not None and len(class_names) > 0:
            detection_classes = []
            for classes_id in results['detection_classes']:
                if classes_id is None:
                    detection_classes.append(None)
                else:
                    detection_classes.append(
                        np.array([class_names[int(id)] for id in classes_id]))
        else:
            detection_classes = results.get('detection_classes', [])

        vis_imgs = []

        img_metas = results['img_metas'][:vis_num]
        detection_boxes = results.get('detection_boxes', [])
        detection_scores = results.get('detection_scores', [])

        for i, img_meta in enumerate(img_metas):
            filename = img_meta['filename']
            bboxes = np.array(
                []) if detection_boxes[i] is None else detection_boxes[i]
            scores = detection_scores[i]
            classes = detection_classes[i]

            if scores is not None and score_thr > 0:
                inds = scores > score_thr
                bboxes = bboxes[inds]
                classes = classes[inds]

            img = load_image(filename)
            vis_img = imshow_bboxes(
                img=img, bboxes=bboxes, labels=classes, show=False)
            vis_imgs.append(vis_img)

        output = {'images': vis_imgs, 'img_metas': img_metas}

        return output
