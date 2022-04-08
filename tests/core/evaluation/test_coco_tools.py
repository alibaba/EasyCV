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
"""Tests for tensorflow_model.metrics.coco_tools."""
import json
import os
import re
import tempfile
import unittest

import numpy as np
from xtcocotools import mask

from easycv.core.evaluation import coco_tools
from easycv.file import io


class CocoToolsTest(unittest.TestCase):

    def setUp(self):
        groundtruth_annotations_list = [
            {
                'id': 1,
                'image_id': 'first',
                'category_id': 1,
                'bbox': [100., 100., 100., 100.],
                'area': 100.**2,
                'iscrowd': 0
            },
            {
                'id': 2,
                'image_id': 'second',
                'category_id': 1,
                'bbox': [50., 50., 50., 50.],
                'area': 50.**2,
                'iscrowd': 0
            },
        ]
        image_list = [{'id': 'first'}, {'id': 'second'}]
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
        self._groundtruth_dict = {
            'annotations': groundtruth_annotations_list,
            'images': image_list,
            'categories': category_list
        }

        self._detections_list = [
            {
                'image_id': 'first',
                'category_id': 1,
                'bbox': [100., 100., 100., 100.],
                'score': .8
            },
            {
                'image_id': 'second',
                'category_id': 1,
                'bbox': [50., 50., 50., 50.],
                'score': .7
            },
        ]

    def testCocoWrappers(self):
        groundtruth = coco_tools.COCOWrapper(self._groundtruth_dict)
        detections = groundtruth.LoadAnnotations(self._detections_list)
        evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections)
        summary_metrics, _ = evaluator.ComputeMetrics()
        self.assertAlmostEqual(1.0, summary_metrics['Precision/mAP'])

    def testExportGroundtruthToCOCO(self):
        image_ids = ['first', 'second']
        groundtruth_boxes = [
            np.array([[100, 100, 200, 200]], np.float),
            np.array([[50, 50, 100, 100]], np.float)
        ]
        groundtruth_classes = [
            np.array([1], np.int32),
            np.array([1], np.int32)
        ]
        categories = [{
            'id': 0,
            'name': 'person'
        }, {
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }]
        output_path = os.path.join(tempfile.gettempdir(), 'groundtruth.json')
        result = coco_tools.ExportGroundtruthToCOCO(
            image_ids,
            groundtruth_boxes,
            groundtruth_classes,
            categories,
            output_path=output_path)
        self.assertDictEqual(result, self._groundtruth_dict)
        with io.open(output_path, 'r') as f:
            written_result = f.read()
            # The json output should have floats written to 4 digits of precision.
            matcher = re.compile(r'"bbox":\s+\[\n\s+\d+.\d\d\d\d,',
                                 re.MULTILINE)
            self.assertTrue(matcher.findall(written_result))
            written_result = json.loads(written_result)
            self.assertAlmostEqual(result, written_result)

    def testExportDetectionsToCOCO(self):
        image_ids = ['first', 'second']
        detections_boxes = [
            np.array([[100, 100, 200, 200]], np.float),
            np.array([[50, 50, 100, 100]], np.float)
        ]
        detections_scores = [
            np.array([.8], np.float),
            np.array([.7], np.float)
        ]
        detections_classes = [np.array([1], np.int32), np.array([1], np.int32)]
        categories = [{
            'id': 0,
            'name': 'person'
        }, {
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }]
        output_path = os.path.join(tempfile.gettempdir(), 'detections.json')
        result = coco_tools.ExportDetectionsToCOCO(
            image_ids,
            detections_boxes,
            detections_scores,
            detections_classes,
            categories,
            output_path=output_path)
        self.assertListEqual(result, self._detections_list)
        with io.open(output_path, 'r') as f:
            written_result = f.read()
            # The json output should have floats written to 4 digits of precision.
            matcher = re.compile(r'"bbox":\s+\[\n\s+\d+.\d\d\d\d,',
                                 re.MULTILINE)
            self.assertTrue(matcher.findall(written_result))
            written_result = json.loads(written_result)
            self.assertAlmostEqual(result, written_result)

    def testExportSegmentsToCOCO(self):
        image_ids = ['first', 'second']
        detection_masks = [
            np.array(
                [[[0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1]]],
                dtype=np.uint8),
            np.array(
                [[[0, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1], [0, 1, 0, 1]]],
                dtype=np.uint8)
        ]

        for i, detection_mask in enumerate(detection_masks):
            detection_masks[i] = detection_mask[:, :, :, None]

        detection_scores = [np.array([.8], np.float), np.array([.7], np.float)]
        detection_classes = [np.array([1], np.int32), np.array([1], np.int32)]

        categories = [{
            'id': 0,
            'name': 'person'
        }, {
            'id': 1,
            'name': 'cat'
        }, {
            'id': 2,
            'name': 'dog'
        }]
        output_path = os.path.join(tempfile.gettempdir(), 'segments.json')
        result = coco_tools.ExportSegmentsToCOCO(
            image_ids,
            detection_masks,
            detection_scores,
            detection_classes,
            categories,
            output_path=output_path)
        with io.open(output_path, 'r') as f:
            written_result = f.read()
            written_result = json.loads(written_result)
            mask_load = mask.decode([written_result[0]['segmentation']])
            self.assertTrue(np.allclose(mask_load, detection_masks[0]))
            self.assertAlmostEqual(result, written_result)

    def testExportKeypointsToCOCO(self):
        image_ids = ['first', 'second']
        detection_keypoints = [
            np.array([[[100, 200], [300, 400], [500, 600]],
                      [[50, 150], [250, 350], [450, 550]]],
                     dtype=np.int32),
            np.array([[[110, 210], [310, 410], [510, 610]],
                      [[60, 160], [260, 360], [460, 560]]],
                     dtype=np.int32)
        ]

        detection_scores = [
            np.array([.8, 0.2], np.float),
            np.array([.7, 0.3], np.float)
        ]
        detection_classes = [
            np.array([1, 1], np.int32),
            np.array([1, 1], np.int32)
        ]

        categories = [{
            'id': 1,
            'name': 'person',
            'num_keypoints': 3
        }, {
            'id': 2,
            'name': 'cat'
        }, {
            'id': 3,
            'name': 'dog'
        }]

        output_path = os.path.join(tempfile.gettempdir(), 'keypoints.json')
        result = coco_tools.ExportKeypointsToCOCO(
            image_ids,
            detection_keypoints,
            detection_scores,
            detection_classes,
            categories,
            output_path=output_path)

        with io.open(output_path, 'r') as f:
            written_result = f.read()
            written_result = json.loads(written_result)
            self.assertAlmostEqual(result, written_result)

    def testSingleImageDetectionBoxesExport(self):
        boxes = np.array([[0, 0, 1, 1], [0, 0, .5, .5], [.5, .5, 1, 1]],
                         dtype=np.float32)
        classes = np.array([1, 2, 3], dtype=np.int32)
        scores = np.array([0.8, 0.2, 0.7], dtype=np.float32)
        coco_boxes = np.array([[0, 0, 1, 1], [0, 0, .5, .5], [.5, .5, .5, .5]],
                              dtype=np.float32)
        coco_annotations = coco_tools.ExportSingleImageDetectionBoxesToCoco(
            image_id='first_image',
            category_id_set=set([1, 2, 3]),
            detection_boxes=boxes,
            detection_classes=classes,
            detection_scores=scores)
        for i, annotation in enumerate(coco_annotations):
            self.assertEqual(annotation['image_id'], 'first_image')
            self.assertEqual(annotation['category_id'], classes[i])
            self.assertAlmostEqual(annotation['score'], scores[i])
            self.assertTrue(
                np.all(np.isclose(annotation['bbox'], coco_boxes[i])))

    def testSingleImageDetectionMaskExport(self):
        masks = np.array(
            [[[1, 1], [1, 1]], [[0, 0], [0, 1]], [[0, 0], [0, 0]]],
            dtype=np.uint8)
        classes = np.array([1, 2, 3], dtype=np.int32)
        scores = np.array([0.8, 0.2, 0.7], dtype=np.float32)
        coco_annotations = coco_tools.ExportSingleImageDetectionMasksToCoco(
            image_id='first_image',
            category_id_set=set([1, 2, 3]),
            detection_classes=classes,
            detection_scores=scores,
            detection_masks=masks)
        expected_counts = ['04', '31', '4']
        for i, mask_annotation in enumerate(coco_annotations):
            self.assertEqual(mask_annotation['segmentation']['counts'],
                             expected_counts[i])
            self.assertTrue(
                np.all(
                    np.equal(
                        mask.decode(mask_annotation['segmentation']),
                        masks[i])))
            self.assertEqual(mask_annotation['image_id'], 'first_image')
            self.assertEqual(mask_annotation['category_id'], classes[i])
            self.assertAlmostEqual(mask_annotation['score'], scores[i])

    def testSingleImageGroundtruthExport(self):
        masks = np.array(
            [[[1, 1], [1, 1]], [[0, 0], [0, 1]], [[0, 0], [0, 0]]],
            dtype=np.uint8)
        boxes = np.array([[0, 0, 1, 1], [0, 0, .5, .5], [.5, .5, 1, 1]],
                         dtype=np.float32)
        coco_boxes = np.array([[0, 0, 1, 1], [0, 0, .5, .5], [.5, .5, .5, .5]],
                              dtype=np.float32)
        classes = np.array([1, 2, 3], dtype=np.int32)
        is_crowd = np.array([0, 1, 0], dtype=np.int32)
        next_annotation_id = 1
        expected_counts = ['04', '31', '4']

        # Tests exporting without passing in is_crowd (for backward compatibility).
        coco_annotations = coco_tools.ExportSingleImageGroundtruthToCoco(
            image_id='first_image',
            category_id_set=set([1, 2, 3]),
            next_annotation_id=next_annotation_id,
            groundtruth_boxes=boxes,
            groundtruth_classes=classes,
            groundtruth_masks=masks)
        for i, annotation in enumerate(coco_annotations):
            self.assertEqual(annotation['segmentation']['counts'],
                             expected_counts[i])
            self.assertTrue(
                np.all(
                    np.equal(
                        mask.decode(annotation['segmentation']), masks[i])))
            self.assertTrue(
                np.all(np.isclose(annotation['bbox'], coco_boxes[i])))
            self.assertEqual(annotation['image_id'], 'first_image')
            self.assertEqual(annotation['category_id'], classes[i])
            self.assertEqual(annotation['id'], i + next_annotation_id)

        # Tests exporting with is_crowd.
        coco_annotations = coco_tools.ExportSingleImageGroundtruthToCoco(
            image_id='first_image',
            category_id_set=set([1, 2, 3]),
            next_annotation_id=next_annotation_id,
            groundtruth_boxes=boxes,
            groundtruth_classes=classes,
            groundtruth_masks=masks,
            groundtruth_is_crowd=is_crowd)
        for i, annotation in enumerate(coco_annotations):
            self.assertEqual(annotation['segmentation']['counts'],
                             expected_counts[i])
            self.assertTrue(
                np.all(
                    np.equal(
                        mask.decode(annotation['segmentation']), masks[i])))
            self.assertTrue(
                np.all(np.isclose(annotation['bbox'], coco_boxes[i])))
            self.assertEqual(annotation['image_id'], 'first_image')
            self.assertEqual(annotation['category_id'], classes[i])
            self.assertEqual(annotation['iscrowd'], is_crowd[i])
            self.assertEqual(annotation['id'], i + next_annotation_id)


if __name__ == '__main__':
    unittest.main()
