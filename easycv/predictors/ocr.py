# Copyright (c) Alibaba, Inc. and its affiliates.
# debug
import sys
sys.path.append('/root/code/ocr/EasyCV')

import cv2
import numpy as np
import torch
import copy
import math
from torchvision.transforms import Compose
from PIL import Image, ImageDraw, ImageFont

from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.models import build_model
from easycv.predictors.builder import PREDICTORS
from easycv.predictors.interface import PredictorInterface
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.registry import build_from_cfg


@PREDICTORS.register_module()
class OCRDetPredictor(PredictorInterface):
    
    def __init__(self, det_model_path):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # build detection model
        self.det_model_path = det_model_path
        self.det_model = None
        with io.open(self.det_model_path, 'rb') as infile:
            det_checkpoint = torch.load(infile, map_location='cpu')
        
        assert 'meta' in det_checkpoint and 'config' in det_checkpoint[
            'meta'], 'meta.config is missing from checkpoint'
        
        self.det_cfg = det_checkpoint['meta']['config']
        #debug
        self.det_cfg['model']['postprocess']['box_thresh']=0.4
        # self.det_cfg['model']['postprocess']['thresh']=0.1
        self.det_cfg['model']['postprocess']['unclip_ratio']=2
        self.det_cfg['model']['postprocess']['score_mode']='slow'
        self.det_model = build_model(self.det_cfg.model)
        self.ckpt = load_checkpoint(
            self.det_model, self.det_model_path, map_location=self.device)
        self.det_model.to(self.device)
        self.det_model.eval()
        
        # build pipeline
        test_pipeline = self.det_cfg.test_pipeline
        pipeline = [build_from_cfg(p, PIPELINES) for p in test_pipeline]
        self.pipeline = Compose(pipeline)
        
    def predict(self, input_data_list):
        """
        Args:
            input_data_list: a list of numpy array(in rgb order), each array is a sample
        to be predicted
        """
        if isinstance(input_data_list,list):
            output_list = []
            for idx, img in enumerate(input_data_list):
                res = self.predict_single(img)
                output_list.append(res[0])
            return output_list
        else:
            res = self.predict_single(input_data_list)[0]
            return res
    
    def predict_single(self, img):
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        ori_shape = img.shape
        data_dict = {'img': img,
                    'ori_img_shape': ori_shape}
        data_dict = self.pipeline(data_dict)
        img = data_dict['img']
        img = torch.unsqueeze(img, 0).to(self.device)
        with torch.no_grad():
            res = self.det_model.extract_feat(img, )
        res = self.det_model.postprocess(res,[ori_shape])
        return res
    
    def show(self, dt_boxes, img):
        # img = img[...,::-1]
        img = img.astype(np.uint8)
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
        return img
    

@PREDICTORS.register_module()
class OCRRecPredictor(PredictorInterface):
    def __init__(self, rec_model_path):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # build detection model
        self.rec_model_path = rec_model_path
        self.rec_model = None
        with io.open(self.rec_model_path, 'rb') as infile:
            rec_checkpoint = torch.load(infile, map_location='cpu')
        
        assert 'meta' in rec_checkpoint and 'config' in rec_checkpoint[
            'meta'], 'meta.config is missing from checkpoint'
        
        self.rec_cfg = rec_checkpoint['meta']['config']
        self.rec_model = build_model(self.rec_cfg.model)
        self.ckpt = load_checkpoint(
            self.rec_model, self.rec_model_path, map_location=self.device)
        self.rec_model.to(self.device)
        self.rec_model.eval()
        
        # build pipeline
        test_pipeline = self.rec_cfg.test_pipeline
        pipeline = [build_from_cfg(p, PIPELINES) for p in test_pipeline]
        self.pipeline = Compose(pipeline)
        
    # def predict(self, input_data_list):
    #     """
    #     Args:
    #         input_data_list: a list of numpy array(in rgb order), each array is a sample
    #     to be predicted
    #     """
    #     if isinstance(input_data_list,list):
    #         output_list = []
    #         for idx, img in enumerate(input_data_list):
    #             res = self.predict_single(img)
    #             output_list.append(res)
    #         return output_list
    #     else:
    #         res = self.predict_single(input_data_list)
    #         return res
    
    def predict(self, input_data_list):
        """
        Args:
            input_data_list: a list of numpy array(in rgb order), each array is a sample
        to be predicted
        """
        if not input_data_list:
            return {'preds_text': []}
        elif isinstance(input_data_list,list):
            img_list = []
            output_list = []
            for idx, img in enumerate(input_data_list):
                img = self.preprocess(img)
                img_list.append(img)
            img_batch = torch.stack(img_list).to(self.device)
            res = self.rec_model.forward_test(img_batch, )
            return res
        else:
            res = self.predict_single(input_data_list)
            return res
    
    def predict_single(self, img):
        
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        ori_shape = img.shape
        data_dict = {'img': img,
                        'ori_shape': ori_shape}
        data_dict = self.pipeline(data_dict)
        img = data_dict['img']
        img = torch.unsqueeze(img, 0).to(self.device)
        res = self.rec_model.forward_test(img, )
        # res = self.rec_model.postprocess(res)
        return res
    
    def preprocess(self, img):
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        ori_shape = img.shape
        data_dict = {'img': img,
                        'ori_shape': ori_shape}
        data_dict = self.pipeline(data_dict)
        img = data_dict['img']
        return img

@PREDICTORS.register_module()
class OCRClsPredictor(PredictorInterface):
    def __init__(self, cls_model_path):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # build detection model
        self.cls_model_path = cls_model_path
        self.cls_model = None
        with io.open(self.cls_model_path, 'rb') as infile:
            cls_checkpoint = torch.load(infile, map_location='cpu')
        
        assert 'meta' in cls_checkpoint and 'config' in cls_checkpoint[
            'meta'], 'meta.config is missing from checkpoint'
        
        self.cls_cfg = cls_checkpoint['meta']['config']
        self.cls_model = build_model(self.cls_cfg.model)
        self.ckpt = load_checkpoint(
            self.cls_model, self.cls_model_path, map_location=self.device)
        self.cls_model.to(self.device)
        self.cls_model.eval()
        
        # build pipeline
        test_pipeline = self.cls_cfg.test_pipeline
        pipeline = [build_from_cfg(p, PIPELINES) for p in test_pipeline]
        self.pipeline = Compose(pipeline)
        
    def predict(self, input_data_list):
        """
        Args:
            input_data_list: a list of numpy array(in rgb order), each array is a sample
        to be predicted
        """
        # if not input_data_list:
        #     return {'class': []}
        if isinstance(input_data_list,list):
            img_list = []
            output_list = []
            for idx, img in enumerate(input_data_list):
                img = self.preprocess(img)
                img_list.append(img)
            img_batch = torch.stack(img_list).to(self.device)
            res = self.cls_model.forward_test(img_batch, )
        else:
            res = self.predict_single(input_data_list)
        res = self.postprocess(res,input_data_list)
        return res
    
    def predict_single(self, img):
        
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        ori_shape = img.shape
        data_dict = {'img': img,
                        'ori_shape': ori_shape}
        data_dict = self.pipeline(data_dict)
        img = data_dict['img']
        img = torch.unsqueeze(img, 0).to(self.device)
        res = self.cls_model.forward_test(img, )
        # res = self.rec_model.postprocess(res)
        return res
    
    def preprocess(self, img):
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        ori_shape = img.shape
        data_dict = {'img': img,
                        'ori_shape': ori_shape}
        data_dict = self.pipeline(data_dict)
        img = data_dict['img']
        return img
    
    def postprocess(self, result, img_list, threshold=0.9):
        labels, logits = result['class'], result['neck']
        result = {'labels':[], 'logits':[]}
        img_list_out = []
        for img, label,logit in zip(img_list, labels, logits):
            result['labels'].append(label)
            result['logits'].append(logit[label])
            if label==1 and logit[label]>threshold:
                img = cv2.flip(img,-1)
            img_list_out.append(img)
        return img_list_out, result
            

@PREDICTORS.register_module()
class OCRPredictor(PredictorInterface):
    
    def __init__(self, det_model_path, rec_model_path, cls_model_path=None, drop_score=0.5, use_angle_cls=False):
        
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
    
    def predict_single(self, img):
        ori_im = img.copy()
        # dt_boxes = self.det_predictor.predict(img[...,::-1])

        dt_boxes = self.det_predictor.predict(img)
        dt_boxes = self.sorted_boxes(dt_boxes)
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            # cv2.imwrite(f"test_rgb_{bno}.jpg",img_crop)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, cls_res = self.cls_predictor.predict(img_crop_list)
                
        for idx, img_crop in enumerate(img_crop_list):
            cv2.imwrite(f"test_rgb_{idx}.jpg",img_crop)
        rec_res = self.rec_predictor.predict(img_crop_list)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res['preds_text']):
            # text, score = rec_reuslt['preds_text'][0]
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                # filter_rec_res.append(rec_reuslt['preds_text'][0])
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res
        
    def show(self, boxes, rec_res, img , drop_score=0.5, font_path="./doc/simfang.ttf"):
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
        draw_img = draw_img[...,::-1]
        return draw_img

def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="./doc/simfang.ttf"):
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
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)

if __name__=="__main__":
    
    # det
    # predictor  = OCRDetPredictor(det_model_path='/root/code/ocr/paddle_to_torch_tools/paddle_weights/ch_ptocr_v3_det_infer_export.pth')
    # predictor  = OCRDetPredictor(det_model_path='/root/code/ocr/EasyCV/out/det_ocr_ch/epoch_1_export.pth')
    # img = cv2.imread('/root/code/ocr/test_img/test_ocr.jpg')
    # # img = cv2.imread('/root/code/ocr/ppocr_img/ch/ch.jpg')
    # # img = img[...,::-1]
    # dt_boxes = predictor.predict(img)
    # src_img = predictor.show(dt_boxes,img)
    # cv2.imwrite("test.jpg",src_img)
    
    # rec
    # predictor = OCRRecPredictor(rec_model_path='../../out/rec_ocr_ch/epoch_10_export.pth')
    # # img = cv2.imread('/root/code/ocr/ppocr_img/ch/word_1.jpg')
    # # img = cv2.imread('test_rgb_3.jpg')
    # import glob
    # img_list = glob.glob('/nas/database/ocr/rec/pai/img/test/*png')
    # sorted(img_list)
    # for img_path in img_list:
    #     img = cv2.imread(img_path)
    #     rec_out = predictor.predict(img)
    #     print(img_path,rec_out['preds_text'][0][0])
    
    
    #cls
    # predictor = OCRClsPredictor(cls_model_path='/root/code/ocr/PaddleOCR/pretrain_models/ch_ppocr_mobile_v2.0_cls_train/best_accuracy_export.pth')
    # img = cv2.imread('/root/code/ocr/ppocr_img/ch/word_1.jpg')
    # cls_out = predictor.predict([img]*4)
    # print(cls_out)
    
    # system
    predictor = OCRPredictor(det_model_path='../../out/det_ocr_ch/epoch_10_export.pth', \
                             rec_model_path='/root/code/ocr/PaddleOCR/pretrain_models/ch_PP-OCRv3_rec_train/best_accuracy_student_export.pth',
                             cls_model_path='/root/code/ocr/PaddleOCR/pretrain_models/ch_ppocr_mobile_v2.0_cls_train/best_accuracy_export.pth',
                             use_angle_cls=True)
    # predictor = OCRPredictor(det_model_path='/root/code/ocr/PaddleOCR/pretrain_models/Multilingual_PP-OCRv3_det_distill_train/student_export.pth', \
    #                          rec_model_path='/root/code/ocr/PaddleOCR/pretrain_models/ch_PP-OCRv3_rec_train/best_accuracy_student_export.pth')
    # img = cv2.imread('/root/code/ocr/test_img/test_ocr.jpg')
    # img = cv2.imread('/root/code/ocr/ppocr_img/ch/ch.jpg')
    # img = cv2.imread('/nas/database/ocr/rec/pai/img/train/217745.png')
    import glob
    import time
    img_list = glob.glob('/nas/database/ocr/det/pai/img/test/*')
    # img_list = glob.glob('/nas/database/ocr/det/icdar2015/text_localization/ch4_test_images/*')
    time_list = []
    img_list = ['/root/code/ocr/test_img/test_ocr.jpg']
    for img_path in img_list:
        img = cv2.imread(img_path)
        tic = time.time()
        filter_boxes, filter_rec_res = predictor.predict_single(img)
        print(time.time()-tic)
        time_list.append(time.time()-tic)
        out_img = predictor.show(filter_boxes, filter_rec_res, img, font_path='/nas/code/ocr/PaddleOCR2Pytorch-main/doc/fonts/simfang.ttf')
        cv2.imwrite('test_paddle/'+img_path.split('/')[-1],out_img)
    
    
    
    
    
    


        
