# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import Compose

from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.models import build_model
from easycv.predictors.builder import PREDICTORS
from easycv.predictors.interface import PredictorInterface
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.registry import build_from_cfg
from .base import PredictorV2


@PREDICTORS.register_module()
class OCRDetPredictor(PredictorV2):

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 *args,
                 **kwargs):

        super(OCRDetPredictor, self).__init__(
            model_path,
            config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            *args,
            **kwargs)

    def preprocess_single(self, input):
        """Preprocess single input sample.
        If you need custom ops to load or process a single input sample, you need to reimplement it.
        """
        input = self._load_input(input)
        input['ori_img_shape'] = input['ori_shape']
        return self.processor(input)

    def forward(self, inputs):
        """Model forward.
        If you need refactor model forward, you need to reimplement it.
        """
        shape_list = [
            img_meta['ori_img_shape'] for img_meta in inputs['img_metas']
        ]
        with torch.no_grad():
            outputs = self.model.extract_feat(inputs['img'], )
        outputs = self.model.postprocess(outputs, shape_list)
        return outputs

    def postprocess(self, inputs, *args, **kwargs):
        return inputs

    def show_result(self, dt_boxes, img):
        img = img.astype(np.uint8)
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
        return img


@PREDICTORS.register_module()
class OCRRecPredictor(PredictorV2):

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 *args,
                 **kwargs):

        super(OCRRecPredictor, self).__init__(
            model_path,
            config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            *args,
            **kwargs)


@PREDICTORS.register_module()
class OCRClsPredictor(PredictorV2):

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 *args,
                 **kwargs):

        super(OCRClsPredictor, self).__init__(
            model_path,
            config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            *args,
            **kwargs)

    def flip_img(self, result, img_list, threshold=0.9):
        output = {'labels': [], 'logits': []}
        img_list_out = []
        for img, res in zip(img_list, result):
            label, logit = res['class'], res['neck']
            output['labels'].append(label)
            output['logits'].append(logit[label])
            if label == 1 and logit[label] > threshold:
                img = cv2.flip(img, -1)
            img_list_out.append(img)
        return img_list_out, output


@PREDICTORS.register_module()
class OCRPredictor(PredictorInterface):

    def __init__(self,
                 det_model_path,
                 rec_model_path,
                 cls_model_path=None,
                 drop_score=0.5,
                 use_angle_cls=False):

        self.use_angle_cls = use_angle_cls
        if use_angle_cls:
            self.cls_predictor = OCRClsPredictor(cls_model_path)
        self.det_predictor = OCRDetPredictor(det_model_path)
        self.rec_predictor = OCRRecPredictor(rec_model_path)
        self.drop_score = drop_score

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """

        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def __call__(self, img):
        ori_im = img.copy()

        dt_boxes = self.det_predictor([img])[0]
        dt_boxes = self.sorted_boxes(dt_boxes)
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            cls_res = self.cls_predictor(img_crop_list)
            img_crop_list, cls_res = self.cls_predictor.flip_img(
                cls_res, img_crop_list)

        rec_res = self.rec_predictor(img_crop_list)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            score = rec_reuslt['preds_text'][1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt['preds_text'])
        return filter_boxes, filter_rec_res

    def show(self,
             boxes,
             rec_res,
             img,
             drop_score=0.5,
             font_path='./doc/simfang.ttf'):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        draw_img = draw_ocr_box_txt(
            img,
            boxes,
            txts,
            scores,
            drop_score=drop_score,
            font_path=font_path)
        draw_img = draw_img[..., ::-1]
        return draw_img


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path='./doc/simfang.ttf'):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon([
            box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1],
            box[3][0], box[3][1]
        ],
                           outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 +
                               (box[0][1] - box[3][1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 +
                              (box[0][1] - box[1][1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text((box[0][0] + 3, cur_y),
                                c,
                                fill=(0, 0, 0),
                                font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding='utf-8')
            draw_right.text([box[0][0], box[0][1]],
                            txt,
                            fill=(0, 0, 0),
                            font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)