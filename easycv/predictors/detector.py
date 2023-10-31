# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from glob import glob

import numpy as np
import torch

from easycv.core.visualization import imshow_bboxes
from easycv.datasets.utils import replace_ImageToTensor
from easycv.file import io
from easycv.models.detection.utils import postprocess
from easycv.thirdparty.mtcnn import FaceDetector
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.misc import deprecated
from .base import InputProcessor, OutputProcessor, PredictorV2
from .builder import PREDICTORS
from .classifier import TorchClassifier

try:
    from easy_vision.python.inference.predictor import PredictorInterface
except Exception:
    from .interface import PredictorInterface


# 将张量转化为ndarray格式
def onnx_to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()


class DetInputProcessor(InputProcessor):

    def build_processor(self):
        if self.pipelines is not None:
            pipelines = self.pipelines
        elif self.cfg is None:
            pipelines = []
        else:
            pipelines = self.cfg.get('test_pipeline', [])

        # for batch inference
        self.pipelines = replace_ImageToTensor(pipelines)

        return super().build_processor()


class DetOutputProcessor(OutputProcessor):

    def __init__(self, score_thresh, classes=None):
        super(DetOutputProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.classes = classes

    def process_single(self, inputs):
        if inputs['detection_scores'] is None or len(
                inputs['detection_scores']) < 1:
            return inputs

        scores = inputs['detection_scores']
        if scores is not None and self.score_thresh > 0:
            keeped_ids = scores > self.score_thresh
            inputs['detection_scores'] = inputs['detection_scores'][keeped_ids]
            inputs['detection_boxes'] = inputs['detection_boxes'][keeped_ids]
            inputs['detection_classes'] = inputs['detection_classes'][
                keeped_ids]

        class_names = []
        for _, classes_id in enumerate(inputs['detection_classes']):
            if classes_id is None:
                class_names.append(None)
            elif self.classes is not None and len(self.classes) > 0:
                class_names.append(self.classes[int(classes_id)])
            else:
                class_names.append(classes_id)

        inputs['detection_class_names'] = class_names

        return inputs


@PREDICTORS.register_module()
class DetectionPredictor(PredictorV2):
    """Generic Detection Predictor, it will filter bbox results by ``score_threshold`` .

    Args:
        model_path (str): Path of model path.
        config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
        batch_size (int): batch size for forward.
        device (str | torch.device): Support str('cuda' or 'cpu') or torch.device, if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        input_processor_threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 score_threshold=0.5,
                 input_processor_threads=8,
                 mode='BGR',
                 *arg,
                 **kwargs):
        super(DetectionPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=input_processor_threads,
            mode=mode)
        self.score_thresh = score_threshold
        self.CLASSES = self.cfg.get('CLASSES', None)

    def get_input_processor(self):
        return DetInputProcessor(
            self.cfg,
            pipelines=self.pipelines,
            batch_size=self.batch_size,
            threads=self.input_processor_threads,
            mode=self.mode)

    def get_output_processor(self):
        return DetOutputProcessor(self.score_thresh, self.CLASSES)

    def visualize(self, img, results, show=False, out_file=None):
        """Only support show one sample now."""
        bboxes = results['detection_boxes']
        labels = results['detection_class_names']
        img = self.input_processor._load_input(img)['img']
        imshow_bboxes(
            img,
            bboxes,
            labels=labels,
            colors='cyan',
            text_color='cyan',
            font_size=18,
            thickness=2,
            font_scale=0.0,
            show=show,
            out_file=out_file)


class _JitProcessorWrapper:

    def __init__(self, processor, device) -> None:
        self.processor = processor
        self.device = device

    def __call__(self, results):
        if self.processor is not None:
            from mmcv.parallel import DataContainer as DC
            outputs = {}
            img = results['img']
            img = torch.from_numpy(img).to(self.device)
            img, img_meta = self.processor(img.unsqueeze(0))  # process batch
            outputs['img'] = DC(
                img.squeeze(0),
                stack=True)  # DC wrapper for collate batch and to device
            outputs['img_metas'] = DC(img_meta, cpu_only=True)
            return outputs
        return results


class YoloXInputProcessor(DetInputProcessor):
    """Input processor for yolox.

    Args:
        cfg (Config): Config instance.
        pipelines (list[dict]): Data pipeline configs.
        batch_size (int): batch size for forward.
        model_type (str): "raw" or "jit" or "blade"
        jit_processor_path (str): File of the saved processing operator of torch jit type.
        device (str | torch.device): Support str('cuda' or 'cpu') or torch.device, if is None, detect device automatically.
        threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
    """

    def __init__(
        self,
        cfg,
        pipelines=None,
        batch_size=1,
        model_type='raw',
        jit_processor_path=None,
        device=None,
        threads=8,
        mode='BGR',
    ):
        self.model_type = model_type
        self.jit_processor_path = jit_processor_path
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__(
            cfg,
            pipelines=pipelines,
            batch_size=batch_size,
            threads=threads,
            mode=mode)

    def build_processor(self):
        self.jit_preprocess = False
        if self.model_type != 'raw':
            if hasattr(self.cfg, 'export'):
                self.jit_preprocess = self.cfg['export'].get(
                    'preprocess_jit', False)

        if self.model_type != 'raw' and self.jit_preprocess:
            # jit or blade model
            processor = None
            if os.path.exists(self.jit_processor_path):
                if self.threads > 1:
                    raise ValueError(
                        'Not support threads>1 for jit processor !')
                # use a preprocess jit model to speed up
                with io.open(self.jit_processor_path, 'rb') as infile:
                    processor = torch.jit.load(infile, self.device)
            return _JitProcessorWrapper(processor, self.device)
        else:
            return super().build_processor()


class YoloXOutputProcessor(DetOutputProcessor):

    def __init__(self,
                 score_thresh=0.5,
                 model_type='raw',
                 test_conf=0.01,
                 nms_thre=0.65,
                 use_trt_efficientnms=False,
                 classes=None):
        super().__init__(score_thresh, classes)
        self.model_type = model_type
        self.test_conf = test_conf
        self.nms_thre = nms_thre
        self.use_trt_efficientnms = use_trt_efficientnms

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

    def process_single(self, inputs):
        det_out = inputs
        img_meta = det_out['img_metas']

        if self.model_type != 'raw':
            results = det_out['results']
            if self.use_trt_efficientnms:
                det_out = {}
                det_out['detection_boxes'] = results[1] / img_meta[
                    'scale_factor'][0]
                det_out['detection_scores'] = results[2]
                det_out['detection_classes'] = results[3]
            else:
                if self.model_type == 'jit':
                    det_out = self.post_assign(
                        results.unsqueeze(0), img_metas=[img_meta])
                else:
                    det_out = self.post_assign(
                        postprocess(
                            results.unsqueeze(0), len(self.classes),
                            self.test_conf, self.nms_thre),
                        img_metas=[img_meta])
            det_out['detection_scores'] = det_out['detection_scores'][0]
            det_out['detection_boxes'] = det_out['detection_boxes'][0]
            det_out['detection_classes'] = det_out['detection_classes'][0]

        resuts = super().process_single(det_out)
        resuts['ori_img_shape'] = list(img_meta['ori_img_shape'][:2])
        return resuts


@PREDICTORS.register_module()
class YoloXPredictor(DetectionPredictor):
    """Detection predictor for Yolox.

    Args:
        model_path (str): Path of model path.
        config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
        batch_size (int): batch size for forward.
        use_trt_efficientnms (bool): Whether used tensorrt efficient nms operation in the saved model.
        device (str | torch.device): Support str('cuda' or 'cpu') or torch.device, if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        max_det (int): Maximum number of detection output boxes.
        score_thresh (float): Score threshold to filter box.
        nms_thresh (float): Nms threshold to filter box.
        input_processor_threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 use_trt_efficientnms=False,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 max_det=100,
                 score_thresh=0.5,
                 nms_thresh=None,
                 test_conf=None,
                 input_processor_threads=8,
                 mode='BGR',
                 model_type=None):
        self.max_det = max_det
        self.use_trt_efficientnms = use_trt_efficientnms
        self.model_type = model_type
        if self.model_type is None:
            if model_path.endswith('jit'):
                self.model_type = 'jit'
            elif model_path.endswith('blade'):
                self.model_type = 'blade'
            elif model_path.endswith('onnx'):
                self.model_type = 'onnx'
            else:
                self.model_type = 'raw'
        assert self.model_type in ['raw', 'jit', 'blade', 'onnx']

        if self.model_type == 'blade' or self.use_trt_efficientnms:
            import torch_blade

        if self.model_type != 'raw' and config_file is None:
            config_file = model_path + '.config.json'

        super(YoloXPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            score_threshold=score_thresh,
            input_processor_threads=input_processor_threads,
            mode=mode)

        self.test_conf = test_conf or self.cfg['model'].get('test_conf', 0.01)
        self.nms_thre = nms_thresh or self.cfg['model'].get('nms_thre', 0.65)
        self.CLASSES = self.cfg.get('CLASSES', None) or self.cfg.get(
            'classes', None)
        assert self.CLASSES is not None
        self.jit_processor_path = '.'.join(
            self.model_path.split('.')[:-1] + ['preprocess'])

    def _build_model(self):
        if self.model_type != 'raw':
            if self.model_type != 'onnx':
                with io.open(self.model_path, 'rb') as infile:
                    model = torch.jit.load(infile, self.device)
            else:
                import onnxruntime
                if onnxruntime.get_device() == 'GPU':
                    model = onnxruntime.InferenceSession(
                        self.model_path, providers=['CUDAExecutionProvider'])
                else:
                    model = onnxruntime.InferenceSession(self.model_path)
        else:
            from easycv.utils.misc import reparameterize_models
            model = super()._build_model()
            model = reparameterize_models(model)
        return model

    def prepare_model(self):
        """Build model from config file by default.
        If the model is not loaded from a configuration file, e.g. torch jit model, you need to reimplement it.
        """
        model = self._build_model()
        if self.model_type != 'onnx':
            model.to(self.device)
            model.eval()
        if self.model_type == 'raw':
            load_checkpoint(model, self.model_path, map_location='cpu')
        return model

    def model_forward(self, inputs):
        """Model forward.
        If you need refactor model forward, you need to reimplement it.
        """
        if self.model_type != 'raw':
            with torch.no_grad():
                if self.model_type != 'onnx':
                    outputs = self.model(inputs['img'])
                else:
                    outputs = self.model.run(
                        None, {
                            self.model.get_inputs()[0].name:
                            onnx_to_numpy(inputs['img'])
                        })[0]
                    outputs = torch.from_numpy(outputs)
                outputs = {'results': outputs}  # convert to dict format
        else:
            outputs = super().model_forward(inputs)

        if 'img_metas' not in outputs:
            outputs['img_metas'] = inputs['img_metas']

        return outputs

    def get_input_processor(self):
        return YoloXInputProcessor(
            self.cfg,
            pipelines=self.pipelines,
            batch_size=self.batch_size,
            model_type=self.model_type,
            jit_processor_path=self.jit_processor_path,
            device=self.device,
            threads=self.input_processor_threads,
            mode=self.mode,
        )

    def get_output_processor(self):
        return YoloXOutputProcessor(
            score_thresh=self.score_thresh,
            model_type=self.model_type,
            test_conf=self.test_conf,
            nms_thre=self.nms_thre,
            use_trt_efficientnms=self.use_trt_efficientnms,
            classes=self.CLASSES)


@deprecated(reason='Please use YoloXPredictor.')
@PREDICTORS.register_module()
class TorchYoloXPredictor(YoloXPredictor):

    def __init__(self,
                 model_path,
                 max_det=100,
                 score_thresh=0.5,
                 use_trt_efficientnms=False,
                 model_config=None,
                 input_processor_threads=8,
                 mode='BGR'):
        """
        Args:
          model_path: model file path
          max_det: maximum number of detection
          score_thresh:  score_thresh to filter box
          model_config: config string for model to init, in json format
        """
        if model_config:
            if isinstance(model_config, str):
                model_config = json.loads(model_config)
        else:
            model_config = {}

        score_thresh = model_config[
            'score_thresh'] if 'score_thresh' in model_config else score_thresh
        super().__init__(
            model_path,
            config_file=None,
            batch_size=1,
            use_trt_efficientnms=use_trt_efficientnms,
            device=None,
            save_results=False,
            save_path=None,
            pipelines=None,
            max_det=max_det,
            score_thresh=score_thresh,
            nms_thresh=None,
            test_conf=None,
            input_processor_threads=input_processor_threads,
            mode=mode)

    def predict(self, input_data_list, batch_size=-1, to_numpy=True):
        return super().__call__(input_data_list)


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
