# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import numpy as np

from easycv.datasets.pose.data_sources.top_down import DatasetInfo
from easycv.datasets.pose.data_sources.wholebody.wholebody_coco_source import \
    WHOLEBODY_COCO_DATASET_INFO
from easycv.datasets.pose.pipelines.transforms import bbox_cs2xyxy
from easycv.predictors.builder import PREDICTORS, build_predictor
from easycv.predictors.detector import TorchYoloXPredictor
from .base import PredictorV2
from .pose_predictor import _box2cs

SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18], [15, 19],
            [16, 20], [16, 21], [16, 22], [91, 92], [92, 93], [93, 94],
            [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
            [100, 101], [101, 102], [102, 103], [91, 104], [104, 105],
            [105, 106], [106, 107], [91, 108], [108, 109], [109, 110],
            [110, 111], [112, 113], [113, 114], [114, 115], [115, 116],
            [112, 117], [117, 118], [118, 119], [119, 120], [112, 121],
            [121, 122], [122, 123], [123, 124], [112, 125], [125, 126],
            [126, 127], [127, 128], [112, 129], [129, 130], [130, 131],
            [131, 132]]


@PREDICTORS.register_module()
class WholeBodyKeypointsPredictor(PredictorV2):
    """WholeBodyKeypointsPredictor

    Attributes:
        model_path: path of keypoint model
        config_file: path or ``Config`` of config file
        detection_model_config: dict of hand detection model predictor config,
                                example like ``dict(type="", model_path="", config_file="", ......)``
        batch_size: batch_size to infer
        save_results: bool
        save_path: path of result image
        bbox_thr: bounding box threshold
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 detection_predictor_config=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 bbox_thr=None,
                 *args,
                 **kwargs):
        super(WholeBodyKeypointsPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            *args,
            **kwargs)
        self.bbox_thr = bbox_thr
        self.dataset_info = DatasetInfo(WHOLEBODY_COCO_DATASET_INFO)
        self.detection_predictor = build_predictor(detection_predictor_config)

    def process_detection_results(self, det_results, cat_id=0):
        """Process det results, and return a list of bboxes.

        Args:
            det_results (list|tuple): det results.
            cat_id (int): category id (default: 0 for human)

        Returns:
            person_results (list): a list of detected bounding boxes
        """
        if isinstance(det_results, tuple):
            det_results = det_results[0]
        elif isinstance(det_results, list):
            det_results = det_results[0]
        else:
            det_results = det_results

        bboxes = det_results['detection_boxes']
        scores = det_results['detection_scores']
        classes = det_results['detection_classes']

        keeped_ids = classes == cat_id
        bboxes = bboxes[keeped_ids]
        scores = scores[keeped_ids]
        classes = classes[keeped_ids]

        person_results = []
        for idx, bbox in enumerate(bboxes):
            person = {}
            bbox = np.append(bbox, scores[idx])
            person['bbox'] = bbox
            person_results.append(person)

        return person_results

    def _load_input(self, input):
        """ load img and convert detection result to topdown style
        """
        outputs = super()._load_input(input)

        box_id = 0
        det_cat_id = 0

        det_results = self.detection_predictor(
            outputs['filename'], keep_inputs=True)
        person_results = self.process_detection_results(
            det_results, det_cat_id)

        # Select bboxes by score threshold
        bboxes = np.array([box['bbox'] for box in person_results])
        if self.bbox_thr is not None:
            assert bboxes.shape[1] == 5
            valid_idx = np.where(bboxes[:, 4] > self.bbox_thr)[0]
            bboxes = bboxes[valid_idx]
            person_results = [person_results[i] for i in valid_idx]

        output_person_info = []
        for person_result in person_results:
            box = person_result['bbox'][:4]
            box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            data = {
                'image_file':
                outputs['filename'],
                'image_id':
                0,
                'rotation':
                0,
                'joints_3d':
                np.zeros((self.cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'joints_3d_visible':
                np.zeros((self.cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'dataset':
                'TopDownCocoWholeBodyDataset',
                'bbox':
                box,
                'bbox_score':
                person_result['bbox'][4:5],
                'bbox_id':
                box_id,  # need to be assigned if batch_size > 1
                'flip_pairs':
                self.dataset_info.flip_pairs,
                'ann_info': {
                    'image_size': np.array(self.cfg.data_cfg['image_size']),
                    'num_joints': self.cfg.data_cfg['num_joints']
                },
                'filename':
                outputs['filename'],
                'img':
                outputs['img'],
                'img_shape':
                outputs['img_shape'],
                'ori_shape':
                outputs['ori_shape'],
                'img_fields':
                outputs['img_fields'],
            }
            box_id += 1
            output_person_info.append(data)

        return output_person_info

    def preprocess_single(self, input):
        results = []
        outputs = self._load_input(input)

        for output in outputs:
            results.append(self.processor(output))
        return results

    def preprocess(self, inputs, *args, **kwargs):
        """Process all inputs list. And collate to batch and put to target device.
        If you need custom ops to load or process a batch samples, you need to reimplement it.
        """
        batch_outputs = []
        for i in inputs:
            for res in self.preprocess_single(i, *args, **kwargs):
                batch_outputs.append(res)

        batch_outputs = self._collate_fn(batch_outputs)
        batch_outputs['img_metas']._data = [[
            i[j] for i in batch_outputs['img_metas']._data
            for j in range(len(i))
        ]]
        batch_outputs = self._to_device(batch_outputs)
        return batch_outputs

    def postprocess(self, inputs, *args, **kwargs):
        output = {}
        output['keypoints'] = inputs['preds'][:, :, :2]
        output['boxes'] = inputs['boxes']
        bbx = output['boxes']
        for i, bbox in enumerate(output['boxes']):
            center, scale = bbox[:2], bbox[2:4]
            output['boxes'][i][:4] = bbox_cs2xyxy(center, scale)
        output['boxes'] = output['boxes'][:, :4]
        return output

    def show_result(self,
                    image_path,
                    keypoints,
                    boxes=None,
                    scale=4,
                    save_path=None):
        """Draw `result` over `img`.

        Args:
            image_path (str): filepath of img
            keypoints (ndarray): N*21*3
        """
        point_color = [120, 225, 240]
        sk_color = [0, 255, 0]
        img = cv2.imread(image_path)
        img = img.copy()
        img_h, img_w = img.shape[:2]

        for kpts in keypoints:
            # point
            for kid, (x, y) in enumerate(kpts):
                cv2.circle(img, (int(x), int(y)), scale, point_color, -1)
            # skeleton
            for sk_id, sk in enumerate(SKELETON):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h):
                    # skip the link that should not be drawn
                    continue
                cv2.line(img, pos1, pos2, sk_color, thickness=1)

        if boxes is not None:
            for bbox in boxes:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

        if save_path is not None:
            cv2.imwrite(save_path, img)
        return img
