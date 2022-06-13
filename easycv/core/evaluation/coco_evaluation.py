# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Class for evaluating object detections with COCO metrics."""
from __future__ import print_function
import json
import os
import tempfile
from collections import OrderedDict, defaultdict

import numpy as np
import six
import torch
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from easycv.core import standard_fields
from easycv.core.evaluation import coco_tools
from easycv.core.post_processing.nms import oks_nms, soft_oks_nms
from easycv.core.standard_fields import DetectionResultFields, InputDataFields
from easycv.utils.json_utils import MyEncoder
from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


@EVALUATORS.register_module
class CocoDetectionEvaluator(Evaluator):
    """Class to evaluate COCO detection metrics."""

    def __init__(self,
                 classes,
                 include_metrics_per_category=False,
                 all_metrics_per_category=False,
                 coco_analyze=False,
                 dataset_name=None,
                 metric_names=['DetectionBoxes_Precision/mAP']):
        """Constructor.

        Args:
            classes: a list of class name
            include_metrics_per_category: If True, include metrics for each category.
            all_metrics_per_category: Whether to include all the summary metrics for
                each category in per_category_ap. Be careful with setting it to true if
                you have more than handful of ∏, because it will pollute your mldash.
            coco_analyze: If True, will analyze the detection result using coco analysis.
            dataset_name: If not None, dataset_name will be inserted to each metric name.
        """

        super(CocoDetectionEvaluator, self).__init__(dataset_name,
                                                     metric_names)
        # _image_ids is a dictionary that maps unique image ids to Booleans which
        # indicate whether a corresponding detection has been added.
        self._image_ids = {}
        self._groundtruth_list = []
        self._detection_boxes_list = []
        self._annotation_id = 1
        self._metrics = None
        self._analyze_images = None
        self._include_metrics_per_category = include_metrics_per_category
        self._all_metrics_per_category = all_metrics_per_category
        self._analyze = coco_analyze
        self._categories = [None] * len(classes)

        # categories: A list of dicts, each of which has the following keys -
        #    'id': (required) an integer id uniquely identifying this category.
        #    'name': (required) string representing category name e.g., 'cat', 'dog'.
        for idx, name in enumerate(classes):
            self._categories[idx] = {'id': idx, 'name': name}
        self._category_id_set = set([cat['id'] for cat in self._categories])

        # for json formatted evaluation
        self.num_classes = len(classes)
        self.class_names = classes
        self.dataset_name = dataset_name
        self.iou_thrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.proposal_nums = (100, 300, 1000)
        self.coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        self.metric_items = [
            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
        ]
        self.metric = 'bbox'
        self.iou_type = 'bbox'

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._image_ids.clear()
        self._groundtruth_list = []
        self._detection_boxes_list = []

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        If the image has already been added, a warning is logged, and groundtruth is
        ignored.

        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary containing

                :InputDataFields.groundtruth_boxes: float32 numpy array of shape
                    [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
                    [ymin, xmin, ymax, xmax] in absolute image coordinates.
                :InputDataFields.groundtruth_classes: integer numpy array of shape
                    [num_boxes] containing 1-indexed groundtruth classes for the boxes.
                    InputDataFields.groundtruth_is_crowd (optional): integer numpy array of
                    shape [num_boxes] containing iscrowd flag for groundtruth boxes.
        """
        if image_id in self._image_ids:
            print(
                'Ignoring ground truth with image id %s since it was '
                '  previously added', image_id)
            return

        groundtruth_is_crowd = groundtruth_dict.get(
            standard_fields.InputDataFields.groundtruth_is_crowd)
        # Drop groundtruth_is_crowd if empty tensor.
        if groundtruth_is_crowd is not None and not groundtruth_is_crowd.shape[
                0]:
            groundtruth_is_crowd = None

        self._groundtruth_list.extend(
            coco_tools.ExportSingleImageGroundtruthToCoco(
                image_id=image_id,
                next_annotation_id=self._annotation_id,
                category_id_set=self._category_id_set,
                groundtruth_boxes=groundtruth_dict[
                    standard_fields.InputDataFields.groundtruth_boxes],
                groundtruth_classes=groundtruth_dict[
                    standard_fields.InputDataFields.groundtruth_classes],
                groundtruth_is_crowd=groundtruth_is_crowd))
        self._annotation_id += groundtruth_dict[
            standard_fields.InputDataFields.groundtruth_boxes].shape[0]
        # Boolean to indicate whether a detection has been added for this image.
        self._image_ids[image_id] = False

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.

        If a detection has already been added for this image id, a warning is
        logged, and the detection is skipped.

        Args:
            image_id: A unique string/integer identifier for the image.
            detections_dict: A dictionary containing

                :DetectionResultFields.detection_boxes: float32 numpy array of shape
                    [num_boxes, 4] containing `num_boxes` detection boxes of the format
                    [ymin, xmin, ymax, xmax] in absolute image coordinates.
                :DetectionResultFields.detection_scores: float32 numpy array of shape
                    [num_boxes] containing detection scores for the boxes.
                :DetectionResultFields.detection_classes: integer numpy array of shape
                    [num_boxes] containing 1-indexed detection classes for the boxes.

        Raises:
            ValueError: If groundtruth for the image_id is not available.
        """
        if image_id not in self._image_ids:
            raise ValueError(
                'Missing groundtruth for image id: {}'.format(image_id))

        if self._image_ids[image_id]:
            print(
                'Ignoring detection with image id %s since it was '
                'previously added', image_id)
            return

        self._detection_boxes_list.extend(
            coco_tools.ExportSingleImageDetectionBoxesToCoco(
                image_id=image_id,
                category_id_set=self._category_id_set,
                detection_boxes=detections_dict[
                    standard_fields.DetectionResultFields.detection_boxes],
                detection_scores=detections_dict[
                    standard_fields.DetectionResultFields.detection_scores],
                detection_classes=detections_dict[
                    standard_fields.DetectionResultFields.detection_classes]))
        self._image_ids[image_id] = True

    def _evaluate(self, analyze=False):
        """Evaluates the detection boxes and returns a dictionary of coco metrics.
        Args:
            analyze:  if set True, will call coco analyze to analyze false positive,
                return result images for each class, this process is very slow.

        Returns:
            A dictionary holding

            1. summary_metrics:
                'DetectionBoxes_Precision/mAP': mean average precision over classes
                    averaged over IOU thresholds ranging from .5 to .95 with .05
                    increments.
                'DetectionBoxes_Precision/mAP@.50IOU': mean average precision at 50% IOU
                'DetectionBoxes_Precision/mAP@.75IOU': mean average precision at 75% IOU
                'DetectionBoxes_Precision/mAP (small)': mean average precision for small
                    objects (area < 32^2 pixels).
                'DetectionBoxes_Precision/mAP (medium)': mean average precision for
                    medium sized objects (32^2 pixels < area < 96^2 pixels).
                'DetectionBoxes_Precision/mAP (large)': mean average precision for large
                    objects (96^2 pixels < area < 10000^2 pixels).
                'DetectionBoxes_Recall/AR@1': average recall with 1 detection.
                'DetectionBoxes_Recall/AR@10': average recall with 10 detections.
                'DetectionBoxes_Recall/AR@100': average recall with 100 detections.
                'DetectionBoxes_Recall/AR@100 (small)': average recall for small objects
                    with 100.
                'DetectionBoxes_Recall/AR@100 (medium)': average recall for medium objects
                    with 100.
                'DetectionBoxes_Recall/AR@100 (large)': average recall for large objects
                    with 100 detections.

            2. per_category_ap: if include_metrics_per_category is True, category
            specific results with keys of the form:
                'Precision mAP ByCategory/category' (without the supercategory part if
                no supercategories exist). For backward compatibility
                'PerformanceByCategory' is included in the output regardless of
                all_metrics_per_category.

        """
        groundtruth_dict = {
            'annotations': self._groundtruth_list,
            'images': [{
                'id': image_id
            } for image_id in self._image_ids],
            'categories': self._categories
        }
        coco_wrapped_groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
        coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(
            self._detection_boxes_list)
        box_evaluator = coco_tools.COCOEvalWrapper(
            coco_wrapped_groundtruth,
            coco_wrapped_detections,
            agnostic_mode=False)
        box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
            include_metrics_per_category=self._include_metrics_per_category,
            all_metrics_per_category=self._all_metrics_per_category)
        box_metrics.update(box_per_category_ap)
        box_metrics = {
            'DetectionBoxes_' + key: value
            for key, value in iter(box_metrics.items())
        }

        if self._analyze or analyze:
            images = box_evaluator.Analyze()
            return box_metrics, images
        else:
            return box_metrics

    def _evaluate_impl(self, prediction_dict, groundtruth_dict):
        '''
        Args:
            prediction_dict:  A dict of k-v pair, each v is a list of
                tensor or numpy array for detection result. A dictionary containing
                :groundtruth_boxes: float32 numpy array of shape
                    [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
                    [ymin, xmin, ymax, xmax] in absolute image coordinates.
                :groundtruth_classes: integer numpy array of shape
                    [num_boxes] containing 1-indexed groundtruth classes for the boxes.
                    InputDataFields.groundtruth_is_crowd (optional): integer numpy array of
                    shape [num_boxes] containing iscrowd flag for groundtruth boxes.
                :img_metas: List of length number of test images,
                        dict of image meta info, containing filename, ori_img_shape, and so on.
            groundtruth_dict: A dict of k-v pair, each v is a list of
                tensor or numpy array for groundtruth info. A dictionary containing
                :DetectionResultFields.detection_boxes: float32 numpy array of shape
                    [num_boxes, 4] containing `num_boxes` detection boxes of the format
                    [ymin, xmin, ymax, xmax] in absolute image coordinates.
                :DetectionResultFields.detection_scores: float32 numpy array of shape
                    [num_boxes] containing detection scores for the boxes.
                :DetectionResultFields.detection_classes: integer numpy array of shape
                    [num_boxes] containing 1-indexed detection classes for the boxes.
                :groundtruth_is_crowd: integer numpy array of
                    shape [num_boxes] containing iscrowd flag for groundtruth boxes.

        Return:
            dict,  each key is metric_name, value is metric value
        '''
        num_images_det = len(
            prediction_dict[DetectionResultFields.detection_boxes])
        num_images_gt = len(
            groundtruth_dict[InputDataFields.groundtruth_boxes])
        assert num_images_det == num_images_gt, 'detection and groundtruth number is not the same'

        groundtruth_is_crowd_list = groundtruth_dict.get(
            'groundtruth_is_crowd', None)
        for idx, (detection_boxes, detection_classes, detection_scores, img_metas, gt_boxes, gt_classes) in \
            enumerate(zip(prediction_dict['detection_boxes'], prediction_dict['detection_classes'],
                          prediction_dict['detection_scores'], prediction_dict['img_metas'],
                          groundtruth_dict['groundtruth_boxes'], groundtruth_dict['groundtruth_classes'])):

            height, width = img_metas[
                'ori_img_shape'][:2]  # height, width, channel
            image_id = img_metas['filename']
            # tensor caculated on other devices is not on device:0, so transfer them
            if isinstance(detection_boxes, torch.Tensor):
                detection_boxes = detection_boxes.cpu().numpy()
                detection_scores = detection_scores.cpu().numpy()
                detection_classes = detection_classes.cpu().numpy().astype(
                    np.int32)

            if groundtruth_is_crowd_list is None:
                groundtruth_is_crowd = None
            else:
                groundtruth_is_crowd = groundtruth_is_crowd_list[idx]

            groundtruth_dict = {
                'groundtruth_boxes': gt_boxes,
                'groundtruth_classes': gt_classes,
                'groundtruth_is_crowd': groundtruth_is_crowd,
            }
            self.add_single_ground_truth_image_info(image_id, groundtruth_dict)
            if detection_classes is None or detection_scores is None:
                detection_classes = np.array([-1])
                detection_scores = np.array([0.0])
                detection_boxes = np.array([[0.0, 0.0, 0.0, 0.0]])

            # add detection info
            detection_dict = {
                'detection_boxes': detection_boxes,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
            }
            self.add_single_detected_image_info(image_id, detection_dict)

        eval_dict = self._evaluate()
        self.clear()
        return eval_dict


def _check_mask_type_and_value(array_name, masks):
    """Checks whether mask dtype is uint8 and the values are either 0 or 1."""
    if masks.dtype != np.uint8:
        raise ValueError('{} must be of type np.uint8. Found {}.'.format(
            array_name, masks.dtype))
    if np.any(np.logical_and(masks != 0, masks != 1)):
        raise ValueError(
            '{} elements can only be either 0 or 1.'.format(array_name))


@EVALUATORS.register_module
class CocoMaskEvaluator(Evaluator):
    """Class to evaluate COCO detection metrics."""

    def __init__(self,
                 classes,
                 include_metrics_per_category=False,
                 dataset_name=None,
                 metric_names=['DetectionMasks_Precision/mAP']):
        """Constructor.

        Args:
            categories: A list of dicts, each of which has the following keys
                :id: (required) an integer id uniquely identifying this category.
                :name: (required) string representing category name e.g., 'cat', 'dog'.
            include_metrics_per_category: If True, include metrics for each category.
        """
        super(CocoMaskEvaluator, self).__init__(dataset_name, metric_names)
        self._image_id_to_mask_shape_map = {}
        self._image_ids_with_detections = set([])
        self._groundtruth_list = []
        self._detection_masks_list = []
        self._annotation_id = 1
        self._include_metrics_per_category = include_metrics_per_category
        self._categories = [None] * len(classes)

        # categories: A list of dicts, each of which has the following keys -
        #    'id': (required) an integer id uniquely identifying this category.
        #    'name': (required) string representing category name e.g., 'cat', 'dog'.
        for idx, name in enumerate(classes):
            self._categories[idx] = {'id': idx, 'name': name}
        self._category_id_set = set([cat['id'] for cat in self._categories])

    def clear(self):
        """Clears the state to prepare for a fresh evaluation."""
        self._image_id_to_mask_shape_map.clear()
        self._image_ids_with_detections.clear()
        self._groundtruth_list = []
        self._detection_masks_list = []

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        """Adds groundtruth for a single image to be used for evaluation.

        If the image has already been added, a warning is logged, and groundtruth is
        ignored.

        Args:
            image_id: A unique string/integer identifier for the image.
            groundtruth_dict: A dictionary containing
                :InputDataFields.groundtruth_boxes: float32 numpy array of shape
                    [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
                    [ymin, xmin, ymax, xmax] in absolute image coordinates.
                :InputDataFields.groundtruth_classes: integer numpy array of shape
                    [num_boxes] containing 1-indexed groundtruth classes for the boxes.
                :InputDataFields.groundtruth_instance_masks: uint8 numpy array of shape
                    [num_boxes, image_height, image_width] containing groundtruth masks
                    corresponding to the boxes. The elements of the array must be in
                    {0, 1}.
        """
        if image_id in self._image_id_to_mask_shape_map:
            print(
                'Ignoring ground truth with image id %s since it was '
                'previously added', image_id)
            return

        groundtruth_instance_masks = groundtruth_dict[
            standard_fields.InputDataFields.groundtruth_instance_masks]
        _check_mask_type_and_value(
            standard_fields.InputDataFields.groundtruth_instance_masks,
            groundtruth_instance_masks)
        self._groundtruth_list.extend(
            coco_tools.ExportSingleImageGroundtruthToCoco(
                image_id=image_id,
                next_annotation_id=self._annotation_id,
                category_id_set=self._category_id_set,
                groundtruth_boxes=groundtruth_dict[
                    standard_fields.InputDataFields.groundtruth_boxes],
                groundtruth_classes=groundtruth_dict[
                    standard_fields.InputDataFields.groundtruth_classes],
                groundtruth_masks=groundtruth_instance_masks,
                groundtruth_is_crowd=groundtruth_dict.get(
                    standard_fields.InputDataFields.groundtruth_is_crowd,
                    None)))
        self._annotation_id += groundtruth_dict[
            standard_fields.InputDataFields.groundtruth_boxes].shape[0]
        self._image_id_to_mask_shape_map[image_id] = groundtruth_dict[
            standard_fields.InputDataFields.groundtruth_instance_masks].shape

    def add_single_detected_image_info(self, image_id, detections_dict):
        """Adds detections for a single image to be used for evaluation.

        If a detection has already been added for this image id, a warning is
        logged, and the detection is skipped.

        Args:
            image_id: A unique string/integer identifier for the image.
            detections_dict: A dictionary containing -
                DetectionResultFields.detection_scores: float32 numpy array of shape
                [num_boxes] containing detection scores for the boxes.
                DetectionResultFields.detection_classes: integer numpy array of shape
                [num_boxes] containing 1-indexed detection classes for the boxes.
                DetectionResultFields.detection_masks: optional uint8 numpy array of
                shape [num_boxes, image_height, image_width] containing instance
                masks corresponding to the boxes. The elements of the array must be
                in {0, 1}.

        Raises:
            ValueError: If groundtruth for the image_id is not available or if
                spatial shapes of groundtruth_instance_masks and detection_masks are
                incompatible.
        """
        if image_id not in self._image_id_to_mask_shape_map:
            raise ValueError(
                'Missing groundtruth for image id: {}'.format(image_id))

        if image_id in self._image_ids_with_detections:
            print(
                'Ignoring detection with image id %s since it was '
                'previously added', image_id)
            return

        groundtruth_masks_shape = self._image_id_to_mask_shape_map[image_id]
        detection_masks = detections_dict[
            standard_fields.DetectionResultFields.detection_masks]
        if (len(detection_masks) and groundtruth_masks_shape[0] != 0
                and groundtruth_masks_shape[1:] != detection_masks.shape[1:]):
            raise ValueError(
                'Spatial shape of groundtruth masks and detection masks '
                'are incompatible: {} vs {}'.format(groundtruth_masks_shape,
                                                    detection_masks.shape))
        _check_mask_type_and_value(
            standard_fields.DetectionResultFields.detection_masks,
            detection_masks)
        self._detection_masks_list.extend(
            coco_tools.ExportSingleImageDetectionMasksToCoco(
                image_id=image_id,
                category_id_set=self._category_id_set,
                detection_masks=detection_masks,
                detection_scores=detections_dict[
                    standard_fields.DetectionResultFields.detection_scores],
                detection_classes=detections_dict[
                    standard_fields.DetectionResultFields.detection_classes]))
        self._image_ids_with_detections.update([image_id])

    def _evaluate(self):
        """Evaluates the detection masks and returns a dictionary of coco metrics.

        Returns:
            A dictionary holding -

            1. summary_metrics:
                'DetectionMasks_Precision/mAP': mean average precision over classes
                    averaged over IOU thresholds ranging from .5 to .95 with .05 increments.
                'DetectionMasks_Precision/mAP@.50IOU': mean average precision at 50% IOU.
                'DetectionMasks_Precision/mAP@.75IOU': mean average precision at 75% IOU.
                'DetectionMasks_Precision/mAP (small)': mean average precision for small
                    objects (area < 32^2 pixels).
                'DetectionMasks_Precision/mAP (medium)': mean average precision for medium
                    sized objects (32^2 pixels < area < 96^2 pixels).
                'DetectionMasks_Precision/mAP (large)': mean average precision for large
                    objects (96^2 pixels < area < 10000^2 pixels).
                'DetectionMasks_Recall/AR@1': average recall with 1 detection.
                'DetectionMasks_Recall/AR@10': average recall with 10 detections.
                'DetectionMasks_Recall/AR@100': average recall with 100 detections.
                'DetectionMasks_Recall/AR@100 (small)': average recall for small objects
                    with 100 detections.
                'DetectionMasks_Recall/AR@100 (medium)': average recall for medium objects
                    with 100 detections.
                'DetectionMasks_Recall/AR@100 (large)': average recall for large objects
                    with 100 detections.

            2. per_category_ap: if include_metrics_per_category is True, category
            specific results with keys of the form:
                'Precision mAP ByCategory/category' (without the supercategory part if
                no supercategories exist). For backward compatibility
                'PerformanceByCategory' is included in the output regardless of
                all_metrics_per_category.
        """
        groundtruth_dict = {
            'annotations':
            self._groundtruth_list,
            'images': [{
                'id': image_id,
                'height': shape[1],
                'width': shape[2]
            } for image_id, shape in six.iteritems(
                self._image_id_to_mask_shape_map)],
            'categories':
            self._categories
        }
        coco_wrapped_groundtruth = coco_tools.COCOWrapper(
            groundtruth_dict, detection_type='segmentation')
        coco_wrapped_detection_masks = coco_wrapped_groundtruth.LoadAnnotations(
            self._detection_masks_list)
        mask_evaluator = coco_tools.COCOEvalWrapper(
            coco_wrapped_groundtruth,
            coco_wrapped_detection_masks,
            agnostic_mode=False,
            iou_type='segm')
        mask_metrics, mask_per_category_ap = mask_evaluator.ComputeMetrics(
            include_metrics_per_category=self._include_metrics_per_category)
        mask_metrics.update(mask_per_category_ap)
        mask_metrics = {
            'DetectionMasks_' + key: value
            for key, value in six.iteritems(mask_metrics)
        }
        return mask_metrics

    def _evaluate_impl(self, prediction_dict, groundtruth_dict):
        """Evaluate with prediction and groundtruth dict

        Args:
            detections_dict: A dictionary containing -
                :DetectionResultFields.detection_scores: float32 numpy array of shape
                    [num_boxes] containing detection scores for the boxes.
                :DetectionResultFields.detection_classes: integer numpy array of shape
                    [num_boxes] containing 1-indexed detection classes for the boxes.
                :DetectionResultFields.detection_masks: mask rle code or optional uint8
                    numpy array of shape [num_boxes, image_height, image_width] containing
                    instance masks corresponding to the boxes. The elements of the array
                    must be in {0, 1}.
                :img_metas: List of length number of test images,
                    dict of image meta info, containing filename, ori_img_shape, and so on.
            groundtruth_dict: A dictionary containing
                :InputDataFields.groundtruth_boxes: float32 numpy array of shape
                    [num_boxes, 4] containing `num_boxes` groundtruth boxes of the format
                    [ymin, xmin, ymax, xmax] in absolute image coordinates.
                :InputDataFields.groundtruth_classes: integer numpy array of shape
                    [num_boxes] containing 1-indexed groundtruth classes for the boxes.
                :InputDataFields.groundtruth_instance_masks: mask rle code or uint8
                    numpy array of shape [num_boxes, image_height, image_width] containing
                    groundtruth masks corresponding to the boxes. The elements of the array
                    must be in {0, 1}.
                :groundtruth_is_crowd: integer numpy array of
                    shape [num_boxes] containing iscrowd flag for groundtruth boxes.
        """
        num_images_det = len(
            prediction_dict[DetectionResultFields.detection_boxes])
        num_images_gt = len(
            groundtruth_dict[InputDataFields.groundtruth_boxes])
        assert num_images_det == num_images_gt, 'detection and groundtruth number is not the same'

        groundtruth_is_crowd_list = groundtruth_dict.get(
            'groundtruth_is_crowd', None)
        for idx, (detection_masks, detection_classes, detection_scores, img_metas, gt_boxes, gt_masks, gt_classes) in \
            enumerate(zip(prediction_dict['detection_masks'], prediction_dict['detection_classes'],
                          prediction_dict['detection_scores'], prediction_dict['img_metas'],
                          groundtruth_dict['groundtruth_boxes'], groundtruth_dict['groundtruth_instance_masks'],
                          groundtruth_dict['groundtruth_classes'])):

            height, width = img_metas[
                'ori_img_shape'][:2]  # height, width, channel
            image_id = img_metas['filename']
            # tensor caculated on other devices is not on device:0, so transfer them
            if isinstance(detection_masks, torch.Tensor):
                detection_masks = detection_masks.cpu().numpy()
                detection_scores = detection_scores.cpu().numpy()
                detection_classes = detection_classes.cpu().numpy().astype(
                    np.int32)

            if groundtruth_is_crowd_list is None:
                groundtruth_is_crowd = None
            else:
                groundtruth_is_crowd = groundtruth_is_crowd_list[idx]

            if len(gt_masks) == 0:
                gt_masks = np.array([], dtype=np.uint8).reshape(
                    (0, height, width))
            else:
                gt_masks = np.array([
                    self._ann_to_mask(mask, height, width) for mask in gt_masks
                ],
                                    dtype=np.uint8)
            groundtruth_dict = {
                'groundtruth_boxes': gt_boxes,
                'groundtruth_instance_masks': gt_masks,
                'groundtruth_classes': gt_classes,
                'groundtruth_is_crowd': groundtruth_is_crowd,
            }
            self.add_single_ground_truth_image_info(image_id, groundtruth_dict)

            detection_masks = np.array([
                self._ann_to_mask(mask, height, width)
                for mask in detection_masks
            ],
                                       dtype=np.uint8)
            # add detection info
            detection_dict = {
                'detection_masks': detection_masks,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
            }
            self.add_single_detected_image_info(image_id, detection_dict)

        eval_dict = self._evaluate()
        self.clear()
        return eval_dict

    def _ann_to_mask(self, segmentation, height, width):
        from xtcocotools import mask as maskUtils
        segm = segmentation
        h = height
        w = width

        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm

        m = maskUtils.decode(rle)
        return m


@EVALUATORS.register_module
class CoCoPoseTopDownEvaluator(Evaluator):
    """Class to evaluate COCO keypoint topdown metrics.
    """

    def __init__(self, dataset_name=None, metric_names=['AP'], **kwargs):
        super().__init__(dataset_name, metric_names)

        self.vis_thr = kwargs.get('vis_thr', 0.2)
        self.oks_thr = kwargs.get('oks_thr', 0.9)
        self.use_nms = kwargs.get('use_nms', True)
        self.soft_nms = kwargs.get('soft_nms', False)

        metric = self.metric_names
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['AP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

    def _evaluate_impl(self, prediction_dict, groundtruth_dict, **kwargs):
        """
        Args:
            prediction_dict
                :preds (np.ndarray[N,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_ids (list[int]): For example, [1, 2, ...]
                :heatmap (np.ndarray[N, K, H, W]): model output heatmap
                :bbox_id (list(int)): bbox_id

            groundtruth_dict (only support coco style)
                :images (list[dict]): dict keywords: 'file_name', 'height', 'width', 'id' and others
                :annotaions (list[dict]): dict keywords: 'num_keypoints', 'keypoints', 'image_id' and others
                :categories (list[dict]): list of category info

        """
        num_joints = kwargs['num_joints']
        class2id = kwargs['class2id']
        sigmas = kwargs.get('sigmas', None)

        image_ids = prediction_dict['image_ids']
        preds = prediction_dict['preds']
        boxes = prediction_dict['boxes']
        bbox_ids = prediction_dict['bbox_ids']

        kpts = defaultdict(list)
        for i, image_id in enumerate(image_ids):
            kpts[image_id].append({
                'keypoints': preds[i],
                'center': boxes[i][0:2],
                'scale': boxes[i][2:4],
                'area': boxes[i][4],
                'score': boxes[i][5],
                'image_id': image_id,
                'bbox_id': bbox_ids[i]
            })
        kpts = self._sort_and_unique_bboxes(kpts)

        # rescoring and oks nms
        valid_kpts = []
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > self.vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.use_nms:
                nms = soft_oks_nms if self.soft_nms else oks_nms
                keep = nms(img_kpts, self.oks_thr, sigmas=sigmas)
                valid_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                valid_kpts.append(img_kpts)

        data_pack = [{
            'cat_id': cat_id,
            'ann_type': 'keypoints',
            'keypoints': valid_kpts
        } for cls, cat_id in class2id.items() if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(
            data_pack[0], num_joints)
        info_str = self._do_python_keypoint_eval(results, groundtruth_dict,
                                                 sigmas)
        name_value = OrderedDict(info_str)

        return name_value

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        for img_id, persons in kpts.items():
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts

    def _coco_keypoint_results_one_category_kernel(self, data_pack,
                                                   num_joints):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1, num_joints * 3)

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point.tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, results, groundtruth, sigmas=None):
        """Keypoint evaluation using COCOAPI."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            groundtruth_file = os.path.join(tmp_dir,
                                            'groundtruth_keypoints.json')
            with open(groundtruth_file, 'w') as f:
                json.dump(groundtruth, f, sort_keys=True, indent=4)
            coco = COCO(groundtruth_file)

            res_file = os.path.join(tmp_dir, 'result_keypoints.json')
            with open(res_file, 'w') as f:
                json.dump(results, f, sort_keys=True, indent=4, cls=MyEncoder)
            coco_det = coco.loadRes(res_file)

        coco_eval = COCOeval(coco, coco_det, 'keypoints', sigmas=sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str


METRICS.register_default_best_metric(CocoDetectionEvaluator,
                                     'DetectionBoxes_Precision/mAP', 'max')
METRICS.register_default_best_metric(CocoMaskEvaluator,
                                     'DetectionMasks_Precision/mAP', 'max')
METRICS.register_default_best_metric(CoCoPoseTopDownEvaluator, 'AP', 'max')
