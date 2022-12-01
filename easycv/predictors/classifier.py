# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import numpy as np
import torch
from PIL import Image, ImageFile

from easycv.file import io
from easycv.framework.errors import ValueError
from easycv.utils.misc import deprecated
from .base import Predictor, PredictorV2
from .builder import PREDICTORS


@PREDICTORS.register_module()
class ClassificationPredictor(PredictorV2):
    """Predictor for classification.
    Args:
        model_path (str): Path of model path.
        config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
        batch_size (int): batch size for forward.
        device (str): Support 'cuda' or 'cpu', if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        topk (int): Return top-k results. Default: 1.
        pil_input (bool): Whether use PIL image. If processor need PIL input, set true, default false.
        label_map_path (str): File path of saving labels list.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 topk=1,
                 pil_input=True,
                 label_map_path=None,
                 *args,
                 **kwargs):
        super(ClassificationPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            *args,
            **kwargs)
        self.topk = topk
        self.pil_input = pil_input

        # Adapt to torchvision transforms which process PIL inputs.
        if self.pil_input:
            self.INPUT_IMAGE_MODE = 'RGB'

        if label_map_path is None:
            if 'CLASSES' in self.cfg:
                class_list = self.cfg.get('CLASSES', [])
            elif 'class_list' in self.cfg:
                class_list = self.cfg.get('class_list', [])
            else:
                class_list = []
        else:
            with io.open(label_map_path, 'r') as f:
                class_list = f.readlines()
        self.label_map = [i.strip() for i in class_list]

    def _load_input(self, input):
        """Load image from file or numpy or PIL object.
        Args:
            input: File path or numpy or PIL object.
        Returns:
           {
                'filename': filename,
                'img': img,
                'img_shape': img_shape,
                'img_fields': ['img']
            }
        """
        if self.pil_input:
            results = {}
            if isinstance(input, str):
                img = Image.open(input)
                if img.mode.upper() != self.INPUT_IMAGE_MODE.upper():
                    img = img.convert(self.INPUT_IMAGE_MODE.upper())
                results['filename'] = input
            else:
                if isinstance(input, np.ndarray):
                    input = Image.fromarray(input)
                # assert isinstance(input, ImageFile.ImageFile)
                img = input
                results['filename'] = None
            results['img'] = img
            results['img_shape'] = img.size
            results['ori_shape'] = img.size
            results['img_fields'] = ['img']
            return results

        return super()._load_input(input)

    def postprocess(self, inputs, *args, **kwargs):
        """Return top-k results."""
        output_prob = inputs['prob'].data.cpu()
        topk_class = torch.topk(output_prob, self.topk).indices.numpy()
        output_prob = output_prob.numpy()
        batch_results = []
        batch_size = output_prob.shape[0]
        for i in range(batch_size):
            result = {'class': np.squeeze(topk_class[i]).tolist()}
            if isinstance(result['class'], int):
                result['class'] = [result['class']]

            if len(self.label_map) > 0:
                result['class_name'] = [
                    self.label_map[i] for i in result['class']
                ]
                result['class_probs'] = {}
                for l_idx, l_name in enumerate(self.label_map):
                    result['class_probs'][l_name] = output_prob[i][l_idx]

            batch_results.append(result)
        return batch_results


try:
    from easy_vision.python.inference.predictor import PredictorInterface
except:
    from .interface import PredictorInterface


@deprecated(reason='Please use ClassificationPredictor.')
@PREDICTORS.register_module()
class TorchClassifier(PredictorInterface):

    def __init__(self,
                 model_path,
                 model_config=None,
                 topk=1,
                 label_map_path=None):
        """
    init model

    Args:
      model_path: model file path
      model_config: config string for model to init, in json format
    """
        self.predictor = Predictor(model_path)
        if 'class_list' not in self.predictor.cfg and label_map_path is None:
            raise ValueError(
                "label_map_path need to be set, when ckpt doesn't contain class_list"
            )

        if label_map_path is None:
            class_list = self.predictor.cfg.get('class_list', [])
            self.label_map = [i.strip() for i in class_list]
        else:
            class_list = open(label_map_path).readlines()
            self.label_map = [i.strip() for i in class_list]
        self.output_name = ['prob', 'class']
        self.topk = topk if topk < len(class_list) else len(class_list)

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

    def predict(self, input_data_list, batch_size=-1):
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
    """
        num_image = len(input_data_list)
        assert len(
            input_data_list) > 0, 'input images should not be an empty list'
        if batch_size > 0:
            num_batches = int(math.ceil(float(num_image) / batch_size))
            image_list = input_data_list
        else:
            num_batches = 1
            batch_size = len(input_data_list)
            image_list = input_data_list

        outputs_list = []
        for batch_idx in range(num_batches):
            batch_image_list = image_list[batch_idx * batch_size:min(
                (batch_idx + 1) * batch_size, len(image_list))]
            image_tensor_list = self.predictor.preprocess(batch_image_list)
            input_data = self.batch(image_tensor_list)
            output_prob = self.predictor.predict_batch(
                input_data, mode='test')['prob'].data.cpu()

            topk_prob = torch.topk(output_prob, self.topk).values.numpy()
            topk_class = torch.topk(output_prob, self.topk).indices.numpy()
            output_prob = output_prob.numpy()

            for idx in range(len(image_tensor_list)):
                single_result = {}
                single_result['class'] = np.squeeze(topk_class[idx]).tolist()
                if isinstance(single_result['class'], int):
                    single_result['class'] = [single_result['class']]
                single_result['class_name'] = [
                    self.label_map[i] for i in single_result['class']
                ]
                single_result['class_probs'] = {}
                for ldx, i in enumerate(self.label_map):
                    single_result['class_probs'][i] = output_prob[idx][ldx]

                outputs_list.append(single_result)

        return outputs_list
