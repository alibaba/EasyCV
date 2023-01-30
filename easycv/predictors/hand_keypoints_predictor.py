# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import mmcv
import numpy as np

from easycv.predictors.builder import PREDICTORS, build_predictor
from ..datasets.pose.data_sources.hand.coco_hand import \
    COCO_WHOLEBODY_HAND_DATASET_INFO
from ..datasets.pose.data_sources.top_down import DatasetInfo
from .base import InputProcessor, OutputProcessor, PredictorV2
from .pose_predictor import _box2cs

HAND_SKELETON = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
                 [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15],
                 [15, 16], [0, 17], [17, 18], [18, 19], [19, 20], [5, 9],
                 [9, 13], [13, 17]]


class HandkptsInputProcessor(InputProcessor):

    def __init__(self,
                 cfg,
                 detection_predictor_config,
                 pipelines=None,
                 batch_size=1,
                 mode='BGR'):
        assert detection_predictor_config is not None, f"{self.__class__.__name__} need 'detection_predictor_config' " \
                                                       f'property to build hand detection model'
        self.detection_predictor = build_predictor(detection_predictor_config)
        self.dataset_info = DatasetInfo(COCO_WHOLEBODY_HAND_DATASET_INFO)
        super().__init__(
            cfg,
            pipelines=pipelines,
            batch_size=batch_size,
            threads=1,
            mode=mode)

    def _load_input(self, input):
        """ load img and convert detection result to topdown style

        Args:
            input (dict):
                {
                    "inputs": image path,
                    "results": {
                        "detection_boxes": B*ndarray(N*4)
                        "detection_scores": B*ndarray(N,)
                        "detection_classes": B*ndarray(N,)
                    }
                }
        """
        image_path = input['inputs']
        data_list = []
        box_id = 0
        det_bbox_result = input['detection_boxes']
        det_bbox_scores = input['detection_scores']
        img = mmcv.imread(image_path, 'color', self.mode)
        for bbox, score in zip(det_bbox_result, det_bbox_scores):
            center, scale = _box2cs(self.cfg.data_cfg['image_size'], bbox)
            # prepare data
            data = {
                'image_file':
                image_path,
                'img':
                img,
                'image_id':
                0,
                'center':
                center,
                'scale':
                scale,
                'bbox_score':
                score,
                'bbox_id':
                box_id,  # need to be assigned if batch_size > 1
                'dataset':
                'coco_wholebody_hand',
                'joints_3d':
                np.zeros((self.cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'joints_3d_visible':
                np.zeros((self.cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'rotation':
                0,
                'flip_pairs':
                self.dataset_info.flip_pairs,
                'ann_info': {
                    'image_size': np.array(self.cfg.data_cfg['image_size']),
                    'num_joints': self.cfg.data_cfg['num_joints']
                }
            }
            data_list.append(data)
            box_id += 1
        return data_list

    def process_single(self, input):
        results = []
        outputs = self._load_input(input)
        for output in outputs:
            results.append(self.processor(output))
        return results

    def __call__(self, inputs):
        """Process all inputs list. And collate to batch and put to target device.
        If you need custom ops to load or process a batch samples, you need to reimplement it.
        """
        # hand det and return source image
        det_results = self.detection_predictor(inputs, keep_inputs=True)

        batch_outputs = []
        for inp in det_results:
            for res in self.process_single(inp):
                batch_outputs.append(res)
        batch_outputs = self._collate_fn(batch_outputs)
        return batch_outputs


class HandkptsOutputProcessor(OutputProcessor):

    def __call__(self, inputs):
        keypoints = inputs['preds']
        boxes = inputs['boxes']
        for i, bbox in enumerate(boxes):
            center, scale = bbox[:2], bbox[2:4]
            boxes[i][:4] = bbox_cs2xyxy(center, scale)
        boxes = boxes[:, :4]
        # TODO: support multi bboxes for a single sample
        assert len(keypoints.shape) == 3
        assert len(boxes.shape) == 2
        batch_outputs = []
        batch_size = keypoints.shape[0]
        keypoints = np.split(keypoints, batch_size)
        boxes = np.split(boxes, batch_size)
        for i in range(batch_size):
            batch_outputs.append({
                'keypoints': keypoints[i],
                'boxes': boxes[i]
            })
        return batch_outputs


@PREDICTORS.register_module()
class HandKeypointsPredictor(PredictorV2):
    """HandKeypointsPredictor

    Attributes:
        model_path: path of keypoint model
        config_file: path or ``Config`` of config file
        detection_model_config: dict of hand detection model predictor config,
                                example like ``dict(type="", model_path="", config_file="", ......)``
        batch_size (int): batch size for forward.
        device (str): Support 'cuda' or 'cpu', if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        input_processor_threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 detection_predictor_config=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 mode='BGR',
                 *args,
                 **kwargs):

        self.detection_predictor_config = detection_predictor_config

        super(HandKeypointsPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=1,
            mode=mode,
            *args,
            **kwargs)

    def get_input_processor(self):
        return HandkptsInputProcessor(
            self.cfg,
            self.detection_predictor_config,
            pipelines=self.pipelines,
            batch_size=self.batch_size,
            mode=self.mode)

    def get_output_processor(self):
        return HandkptsOutputProcessor()

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
        img = mmcv.imread(image_path)
        img = img.copy()
        img_h, img_w = img.shape[:2]

        for kpts in keypoints:
            # point
            for kid, (x, y, s) in enumerate(kpts):
                cv2.circle(img, (int(x), int(y)), scale, point_color, -1)
            # skeleton
            for sk_id, sk in enumerate(HAND_SKELETON):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h):
                    # skip the link that should not be drawn
                    continue
                cv2.line(img, pos1, pos2, sk_color, thickness=1)

        if boxes is not None:
            bboxes = np.vstack(boxes)
            mmcv.imshow_bboxes(
                img, bboxes, colors='green', top_k=-1, thickness=2, show=False)

        if save_path is not None:
            mmcv.imwrite(img, save_path)
        return img


def bbox_cs2xyxy(center, scale, padding=1., pixel_std=200.):
    wh = scale * 0.8 / padding * pixel_std
    xy = center - 0.5 * wh
    x1, y1 = xy
    w, h = wh
    return np.r_[x1, y1, x1 + w, y1 + h]
