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
"""Tests for tensorflow_models.metrics.coco_evaluation."""

from __future__ import absolute_import, division, print_function
import unittest

import numpy as np

from easycv.core import standard_fields
from easycv.core.evaluation import coco_evaluation


class CocoDetectionEvaluationTest(unittest.TestCase):

    def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
        """Tests that mAP is calculated correctly on GT and Detections."""
        category_list = [{
            'id': 0,
            'name': 'person'
        }, {
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }]
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(category_list)
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image1',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image1',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
            })
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image2',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[50., 50., 100., 100.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image2',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[50., 50., 100., 100.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
            })
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image3',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[25., 25., 50., 50.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image3',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[25., 25., 50., 50.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
            })
        metrics = coco_evaluator._evaluate()
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsSkipCrowd(self):
        """Tests computing mAP with is_crowd GT boxes skipped."""
        category_list = [{
            'id': 0,
            'name': 'person'
        }, {
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }]
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(category_list)
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image1',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.], [99., 99., 200., 200.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1, 2]),
                standard_fields.InputDataFields.groundtruth_is_crowd:
                np.array([0, 1])
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image1',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
            })
        metrics = coco_evaluator._evaluate()
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsEmptyCrowd(self):
        """Tests computing mAP with empty is_crowd array passed in."""
        category_list = [{
            'id': 0,
            'name': 'person'
        }, {
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }]
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(category_list)
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image1',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
                standard_fields.InputDataFields.groundtruth_is_crowd:
                np.array([])
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image1',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
            })
        metrics = coco_evaluator._evaluate()
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

    def testRejectionOnDuplicateGroundtruth(self):
        """Tests that groundtruth cannot be added more than once for an image."""
        categories = [{
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }, {
            'id': 3,
            'name': 'elephant'
        }]
        #  Add groundtruth
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(categories)
        image_key1 = 'img1'
        groundtruth_boxes1 = np.array(
            [[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], dtype=float)
        groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
        coco_evaluator.add_single_ground_truth_image_info(
            image_key1, {
                standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_boxes1,
                standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_labels1
            })
        groundtruth_lists_len = len(coco_evaluator._groundtruth_list)

        # Add groundtruth with the same image id.
        coco_evaluator.add_single_ground_truth_image_info(
            image_key1, {
                standard_fields.InputDataFields.groundtruth_boxes:
                groundtruth_boxes1,
                standard_fields.InputDataFields.groundtruth_classes:
                groundtruth_class_labels1
            })
        self.assertEqual(groundtruth_lists_len,
                         len(coco_evaluator._groundtruth_list))

    def testRejectionOnDuplicateDetections(self):
        """Tests that detections cannot be added more than once for an image."""
        categories = [{
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }, {
            'id': 3,
            'name': 'elephant'
        }]
        #  Add groundtruth
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(categories)
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image1',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[99., 100., 200., 200.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1])
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image1',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
            })
        detections_lists_len = len(coco_evaluator._detection_boxes_list)
        coco_evaluator.add_single_detected_image_info(
            image_id='image1',  # Note that this image id was previously added.
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1])
            })
        self.assertEqual(detections_lists_len,
                         len(coco_evaluator._detection_boxes_list))

    def testExceptionRaisedWithMissingGroundtruth(self):
        """Tests that exception is raised for detection with missing groundtruth."""
        categories = [{
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }, {
            'id': 3,
            'name': 'elephant'
        }]
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(categories)
        with self.assertRaises(ValueError):
            coco_evaluator.add_single_detected_image_info(
                image_id='image1',
                detections_dict={
                    standard_fields.DetectionResultFields.detection_boxes:
                    np.array([[100., 100., 200., 200.]]),
                    standard_fields.DetectionResultFields.detection_scores:
                    np.array([.8]),
                    standard_fields.DetectionResultFields.detection_classes:
                    np.array([1])
                })

    def testEval(self):
        """Tests that mAP is calculated correctly on GT and Detections."""
        category_list = [{
            'id': 0,
            'name': 'person'
        }, {
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }]
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(category_list)
        gt_dict = dict(
            image_id=['image1', 'image2', 'image3'],
            groundtruth_boxes=np.array([[[100., 100., 200., 200.]],
                                        [[50., 50., 100., 100.]],
                                        [[25., 25., 50., 50.]]]),
            groundtruth_classes=np.array([[1], [1], [1]]))
        det_dict = dict(
            img_metas=[
                {
                    'filename': 'image1',
                    'ori_img_shape': [1000, 1000]
                },
                {
                    'filename': 'image2',
                    'ori_img_shape': [1000, 1000]
                },
                {
                    'filename': 'image3',
                    'ori_img_shape': [1000, 1000]
                },
            ],
            # x1, y1, x2, y2 in absolute coords
            detection_boxes=np.array([[[100., 100., 200., 200.]],
                                      [[50., 50., 100., 100.]],
                                      [[25., 25., 50., 50.]]]),
            detection_classes=np.array([[1], [1], [1]]),
            detection_scores=np.array([[0.8], [0.8], [0.8]]))

        eval_res = coco_evaluator.evaluate(det_dict, gt_dict)
        print(eval_res)
        self.assertAlmostEqual(eval_res['DetectionBoxes_Precision/mAP'], 1.0)


class CocoMaskEvaluationTest(unittest.TestCase):

    def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
        category_list = [{
            'id': 0,
            'name': 'person'
        }, {
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }]
        coco_evaluator = coco_evaluation.CocoMaskEvaluator(category_list)
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image1',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
                standard_fields.InputDataFields.groundtruth_instance_masks:
                np.pad(
                    np.ones([1, 100, 100], dtype=np.uint8),
                    ((0, 0), (10, 10), (10, 10)),
                    mode='constant')
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image1',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[100., 100., 200., 200.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
                standard_fields.DetectionResultFields.detection_masks:
                np.pad(
                    np.ones([1, 100, 100], dtype=np.uint8),
                    ((0, 0), (10, 10), (10, 10)),
                    mode='constant')
            })
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image2',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[50., 50., 100., 100.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
                standard_fields.InputDataFields.groundtruth_instance_masks:
                np.pad(
                    np.ones([1, 50, 50], dtype=np.uint8),
                    ((0, 0), (10, 10), (10, 10)),
                    mode='constant')
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image2',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[50., 50., 100., 100.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
                standard_fields.DetectionResultFields.detection_masks:
                np.pad(
                    np.ones([1, 50, 50], dtype=np.uint8),
                    ((0, 0), (10, 10), (10, 10)),
                    mode='constant')
            })
        coco_evaluator.add_single_ground_truth_image_info(
            image_id='image3',
            groundtruth_dict={
                standard_fields.InputDataFields.groundtruth_boxes:
                np.array([[25., 25., 50., 50.]]),
                standard_fields.InputDataFields.groundtruth_classes:
                np.array([1]),
                standard_fields.InputDataFields.groundtruth_instance_masks:
                np.pad(
                    np.ones([1, 25, 25], dtype=np.uint8),
                    ((0, 0), (10, 10), (10, 10)),
                    mode='constant')
            })
        coco_evaluator.add_single_detected_image_info(
            image_id='image3',
            detections_dict={
                standard_fields.DetectionResultFields.detection_boxes:
                np.array([[25., 25., 50., 50.]]),
                standard_fields.DetectionResultFields.detection_scores:
                np.array([.8]),
                standard_fields.DetectionResultFields.detection_classes:
                np.array([1]),
                standard_fields.DetectionResultFields.detection_masks:
                np.pad(
                    np.ones([1, 25, 25], dtype=np.uint8),
                    ((0, 0), (10, 10), (10, 10)),
                    mode='constant')
            })
        metrics = coco_evaluator._evaluate()
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
        coco_evaluator.clear()
        self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
        self.assertFalse(coco_evaluator._image_ids_with_detections)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._detection_masks_list)


if __name__ == '__main__':
    unittest.main()
