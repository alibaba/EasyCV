# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from glob import glob

import cv2
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from torch.hub import load_state_dict_from_url
from torchvision.transforms import Compose

from easycv.core.visualization import imshow_bboxes
from easycv.datasets.registry import PIPELINES
from easycv.datasets.utils import replace_ImageToTensor
from easycv.file import io
from easycv.file.utils import is_url_path, url_path_exists
from easycv.models import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.constant import CACHE_DIR
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab
from easycv.utils.registry import build_from_cfg
from .builder import PREDICTORS
from .classifier import TorchClassifier

try:
    from easy_vision.python.inference.predictor import PredictorInterface
except Exception:
    from .interface import PredictorInterface

try:
    from thirdparty.mtcnn import FaceDetector
except Exception:
    from easycv.thirdparty.mtcnn import FaceDetector


@PREDICTORS.register_module()
class TorchYoloXPredictor(PredictorInterface):

    def __init__(self,
                 model_path,
                 max_det=100,
                 score_thresh=0.5,
                 model_config=None):
        """
        init model

        Args:
          model_path: model file path
          max_det: maximum number of detection
          score_thresh:  score_thresh to filter box
          model_config: config string for model to init, in json format
        """
        self.model_path = model_path
        self.max_det = max_det
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_jit = model_path.endswith('jit') or model_path.endswith(
            'blade')

        self.use_blade = model_path.endswith('blade')

        if self.use_blade:
            import torch_blade

        if model_config:
            model_config = json.loads(model_config)
        else:
            model_config = {}

        self.score_thresh = model_config[
            'score_thresh'] if 'score_thresh' in model_config else score_thresh

        if self.use_jit:
            with io.open(model_path, 'rb') as infile:
                map_location = 'cpu' if self.device == 'cpu' else 'cuda'
                self.model = torch.jit.load(infile, map_location)

            with io.open(model_path + '.config.json', 'r') as infile:
                self.cfg = json.load(infile)
                test_pipeline = self.cfg['test_pipeline']
                self.CLASSES = self.cfg['classes']
                self.end2end = self.cfg['export']['end2end']

            self.traceable = True

        else:
            self.end2end = False
            with io.open(self.model_path, 'rb') as infile:
                checkpoint = torch.load(infile, map_location='cpu')

            assert 'meta' in checkpoint and 'config' in checkpoint[
                'meta'], 'meta.config is missing from checkpoint'

            config_str = checkpoint['meta']['config']
            config_str = config_str[config_str.find('_base_'):]
            # get config
            basename = os.path.basename(self.model_path)
            fname, _ = os.path.splitext(basename)
            self.local_config_file = os.path.join(CACHE_DIR,
                                                  f'{fname}_config.py')
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
            with open(self.local_config_file, 'w') as ofile:
                ofile.write(config_str)

            self.cfg = mmcv_config_fromfile(self.local_config_file)

            # build model
            self.model = build_model(self.cfg.model)
            self.traceable = getattr(self.model, 'trace_able', False)

            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            map_location = 'cpu' if self.device == 'cpu' else 'cuda'
            self.ckpt = load_checkpoint(
                self.model, self.model_path, map_location=map_location)

            self.model.to(self.device)
            self.model.eval()

            test_pipeline = self.cfg.test_pipeline
            self.CLASSES = self.cfg.CLASSES

        # build pipeline
        pipeline = [build_from_cfg(p, PIPELINES) for p in test_pipeline]
        self.pipeline = Compose(pipeline)

    def post_assign(self, outputs, img_metas):
        detection_boxes = []
        detection_scores = []
        detection_classes = []
        img_metas_list = []

        for i in range(len(outputs)):
            if img_metas:
                img_metas_list.append(img_metas[i])
            if outputs[i].requires_grad == True:
                outputs[i] = outputs[i].detach()
            if outputs[i] is not None:
                bboxes = outputs[i][:, 0:4] if outputs[i] is not None else None
                if img_metas:
                    bboxes /= img_metas[i]['scale_factor'][0]
                detection_boxes.append(bboxes.cpu().numpy())
                detection_scores.append(
                    (outputs[i][:, 4] * outputs[i][:, 5]).cpu().numpy())
                detection_classes.append(outputs[i][:, 6].cpu().numpy().astype(
                    np.int32))
            else:
                detection_boxes.append(None)
                detection_scores.append(None)
                detection_classes.append(None)

        test_outputs = {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
            'img_metas': img_metas_list
        }
        return test_outputs

    def predict(self, input_data_list, batch_size=-1, to_numpy=True):
        """
    using session run predict a number of samples using batch_size

    Args:
      input_data_list:  a list of numpy array(in rgb order), each array is a sample
        to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    """
        output_list = []
        for idx, img in enumerate(input_data_list):
            if type(img) is not np.ndarray:
                img = np.asarray(img)

            ori_img_shape = img.shape[:2]

            if self.end2end:
                # the input should also be as the type of uint8 as mmcv
                img = torch.from_numpy(img).to(self.device)
                det_out = self.model(img)

                detection_scores = det_out['detection_scores']

                if detection_scores is not None:
                    sel_ids = detection_scores > self.score_thresh
                    detection_scores = detection_scores[sel_ids]
                    detection_boxes = det_out['detection_boxes'][sel_ids]
                    detection_classes = det_out['detection_classes'][sel_ids]
                else:
                    detection_boxes = []
                    detection_classes = []

                if to_numpy:
                    detection_scores = detection_scores.detach().numpy()
                    detection_boxes = detection_boxes.detach().numpy()
                    detection_classes = detection_classes.detach().numpy()

            else:
                data_dict = {'img': img}
                data_dict = self.pipeline(data_dict)
                img = data_dict['img']
                img = torch.unsqueeze(img._data, 0).to(self.device)
                data_dict.pop('img')

                if self.traceable:
                    with torch.no_grad():
                        det_out = self.post_assign(
                            self.model(img),
                            img_metas=[data_dict['img_metas']._data])
                else:
                    with torch.no_grad():
                        det_out = self.model(
                            img,
                            mode='test',
                            img_metas=[data_dict['img_metas']._data])

                # det_out = det_out[:self.max_det]
                # scale box to original image scale, this logic has some operation
                # that can not be traced, see
                # https://discuss.pytorch.org/t/windows-libtorch-c-load-cuda-module-with-std-runtime-error-message-shape-4-is-invalid-for-input-if-size-40/63073/4
                # det_out = scale_coords(img.shape[2:], det_out, ori_img_shape, (scale_factor, pad))

                detection_scores = det_out['detection_scores'][0]

                if detection_scores is not None:
                    sel_ids = detection_scores > self.score_thresh
                    detection_scores = detection_scores[sel_ids]
                    detection_boxes = det_out['detection_boxes'][0][sel_ids]
                    detection_classes = det_out['detection_classes'][0][
                        sel_ids]
                else:
                    detection_boxes = None
                    detection_classes = None

            num_boxes = detection_classes.shape[
                0] if detection_classes is not None else 0

            detection_classes_names = [
                self.CLASSES[detection_classes[idx]]
                for idx in range(num_boxes)
            ]

            out = {
                'ori_img_shape': list(ori_img_shape),
                'detection_boxes': detection_boxes,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
                'detection_class_names': detection_classes_names,
            }

            output_list.append(out)

        return output_list


@PREDICTORS.register_module()
class TorchViTDetPredictor(PredictorInterface):

    def __init__(self, model_path):

        self.model_path = model_path

        if is_url_path(self.model_path) and url_path_exists(self.model_path):
            checkpoint = load_state_dict_from_url(model_path)
        else:
            assert io.exists(
                self.model_path), f'{self.model_path} does not exists'

            with io.open(self.model_path, 'rb') as infile:
                checkpoint = torch.load(infile, map_location='cpu')

        assert 'meta' in checkpoint and 'config' in checkpoint[
            'meta'], 'meta.config is missing from checkpoint'

        config_str = checkpoint['meta']['config']
        if isinstance(config_str, dict):
            config_str = json.dumps(config_str)

        # get config
        basename = os.path.basename(self.model_path)
        fname, _ = os.path.splitext(basename)
        self.local_config_file = os.path.join(CACHE_DIR,
                                              f'{fname}_config.json')
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        with open(self.local_config_file, 'w') as ofile:
            ofile.write(config_str)
        self.cfg = mmcv_config_fromfile(self.local_config_file)

        # dynamic adapt mmdet models
        dynamic_adapt_for_mmlab(self.cfg)

        # build model
        self.model = build_model(self.cfg.model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        map_location = 'cpu' if self.device == 'cpu' else 'cuda'
        self.ckpt = load_checkpoint(
            self.model, self.model_path, map_location=map_location)

        self.model.to(self.device)
        self.model.eval()

        self.CLASSES = self.cfg.CLASSES

    def predict(self, imgs):
        """Inference image(s) with the detector.
        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        Returns:
            If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
        """

        if isinstance(imgs, (list, tuple)):
            is_batch = True
        else:
            imgs = [imgs]
            is_batch = False

        cfg = self.cfg
        device = next(self.model.parameters()).device  # model device

        if isinstance(imgs[0], np.ndarray):
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.val.pipeline.insert(
                0,
                dict(
                    type='LoadImageFromWebcam',
                    file_client_args=dict(backend='http')))
        else:
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.val.pipeline.insert(
                0,
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='http')))

        cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)

        transforms = []
        for transform in cfg.data.val.pipeline:
            if 'img_scale' in transform:
                transform['img_scale'] = tuple(transform['img_scale'])
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                transforms.append(transform)
            elif callable(transform):
                transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')
        test_pipeline = Compose(transforms)

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            results = self.model(mode='test', **data)

        return results

    def show_result_pyplot(self,
                           img,
                           results,
                           score_thr=0.3,
                           show=False,
                           out_file=None):
        bboxes = results['detection_boxes'][0]
        scores = results['detection_scores'][0]
        labels = results['detection_classes'][0].tolist()

        # If self.CLASSES is not None, class_id will be converted to self.CLASSES for visualization,
        # otherwise the class_id will be displayed.
        # And don't try to modify the value in results, it may cause some bugs or even precision problems,
        # because `self.evaluate` will also use the results, refer to: https://github.com/alibaba/EasyCV/pull/67

        if self.CLASSES is not None and len(self.CLASSES) > 0:
            for i, classes_id in enumerate(labels):
                if classes_id is None:
                    labels[i] = None
                else:
                    labels[i] = self.CLASSES[int(classes_id)]

        if scores is not None and score_thr > 0:
            inds = scores > score_thr
            bboxes = bboxes[inds]
            labels = np.array(labels)[inds]

        imshow_bboxes(
            img,
            bboxes,
            labels=labels,
            colors='green',
            text_color='white',
            font_size=20,
            thickness=1,
            font_scale=0.5,
            show=show,
            out_file=out_file)


@PREDICTORS.register_module()
class TorchFaceDetector(PredictorInterface):

    def __init__(self, model_path=None, model_config=None):
        """
    init model, add a facedetect and align for img input.

    Args:
      model_path: model file path
      model_config: config string for model to init, in json format
    """

        self.detector = FaceDetector()

    def get_output_type(self):
        """
    in this function user should return a type dict, which indicates
    which type of data should the output of predictor be converted to
    * type json, data will be serialized to json str

    * type image, data will be converted to encode image binary and write to oss file,
      whose name is output_dir/${key}/${input_filename}_${idx}.jpg, where input_filename
      is the base filename extracted from url, key corresponds to the key in the dict of output_type,
      if the type of data indexed by key is a list, idx is the index of element in list, otherwhile ${idx} will be empty

    * type video, data will be converted to encode video binary and write to oss file,

    :: return  {
      'image': 'image',
      'feature': 'json'
    }
    indicating that the image data in the output dict will be save to image
    file and feature in output dict will be converted to json

    """
        return {}

    def batch(self, image_tensor_list):
        return torch.stack(image_tensor_list)

    def predict(self, input_data_list, batch_size=-1, threshold=0.95):
        """
    using session run predict a number of samples using batch_size

    Args:
      input_data_list:  a list of numpy array, each array is a sample to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    Raise:
       if detect !=1 face in a img, then do nothing for this image
    """
        num_image = len(input_data_list)
        assert len(
            input_data_list) > 0, 'input images should not be an empty list'

        image_list = input_data_list
        outputs_list = []

        for idx, img in enumerate(image_list):
            if type(img) is not np.ndarray:
                img = np.asarray(img)

            ori_img_shape = img.shape[:2]
            bbox, ld = self.detector.safe_detect(img)
            _scores = np.array([i[-1] for i in bbox])

            boxes = []
            scores = []
            for idx, s in enumerate(_scores):
                if s > threshold:
                    boxes.append(bbox[idx][:-1])
                    scores.append(bbox[idx][-1])
            boxes = np.array(boxes)
            scores = np.array(scores)

            out = {
                'ori_img_shape': list(ori_img_shape),
                'detection_boxes': boxes,
                'detection_scores': scores,
                'detection_classes': [0] * boxes.shape[0],
                'detection_class_names': ['face'] * boxes.shape[0],
            }

            outputs_list.append(out)

        return outputs_list


@PREDICTORS.register_module()
class TorchYoloXClassifierPredictor(PredictorInterface):

    def __init__(self,
                 models_root_dir,
                 max_det=100,
                 cls_score_thresh=0.01,
                 det_model_config=None,
                 cls_model_config=None):
        """
    init model, add a yolox and classification predictor for img input.

    Args:
      models_root_dir: models_root_dir/detection/*.pth and models_root_dir/classification/*.pth
      det_model_config: config string for detection model to init, in json format
      cls_model_config: config string for classification model to init, in json format
    """
        det_model_path = glob(
            '%s/detection/*.pt*' % models_root_dir, recursive=True)
        assert (len(det_model_path) == 1)
        cls_model_path = glob(
            '%s/classification/*.pt*' % models_root_dir, recursive=True)
        assert (len(cls_model_path) == 1)

        self.det_predictor = TorchYoloXPredictor(
            det_model_path[0], max_det=max_det, model_config=det_model_config)
        self.cls_predictor = TorchClassifier(
            cls_model_path[0], model_config=cls_model_config)
        self.cls_score_thresh = cls_score_thresh

    def predict(self, input_data_list, batch_size=-1):
        """
    using session run predict a number of samples using batch_size

    Args:
      input_data_list:  a list of numpy array(in rgb order), each array is a sample
        to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    """
        results = self.det_predictor.predict(
            input_data_list, batch_size=batch_size)

        for img_idx, img in enumerate(input_data_list):
            detection_boxes = results[img_idx]['detection_boxes']
            detection_classes = results[img_idx]['detection_classes']
            detection_scores = results[img_idx]['detection_scores']

            crop_img_batch = []
            for idx in range(detection_boxes.shape[0]):
                xyxy = [int(a) for a in detection_boxes[idx]]
                cropImg = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                crop_img_batch.append(cropImg)

            if len(crop_img_batch) > 0:
                cls_output = self.cls_predictor.predict(
                    crop_img_batch, batch_size=32)
            else:
                cls_output = []

            class_name_list = []
            class_id_list = []
            class_score_list = []
            det_bboxes = []
            product_count_dict = {}

            for idx in range(len(cls_output)):
                class_name = cls_output[idx]['class_name'][0]
                class_score = cls_output[idx]['class_probs'][class_name]
                if class_score < self.cls_score_thresh:
                    continue

                if class_name not in product_count_dict:
                    product_count_dict[class_name] = 1
                else:
                    product_count_dict[class_name] += 1
                class_name_list.append(class_name)
                class_id_list.append(int(cls_output[idx]['class'][0]))
                class_score_list.append(class_score)
                det_bboxes.append([float(a) for a in detection_boxes[idx]])

            results[img_idx].update({
                'detection_boxes': np.array(det_bboxes),
                'detection_scores': class_score_list,
                'detection_classes': class_id_list,
                'detection_class_names': class_name_list,
                'product_count': product_count_dict
            })

        return results
