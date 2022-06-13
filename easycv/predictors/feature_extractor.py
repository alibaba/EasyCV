# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from glob import glob

import numpy as np
import torch
from PIL import Image

from .base import Predictor
from .builder import PREDICTORS

try:
    from easy_vision.python.inference.predictor import PredictorInterface
except:
    from .interface import PredictorInterface

try:
    from thirdparty.mtcnn import FaceDetector
    from thirdparty.face_align import glint360k_align
except:
    from easycv.thirdparty.mtcnn import FaceDetector
    from easycv.thirdparty.face_align import glint360k_align


@PREDICTORS.register_module()
class TorchFeatureExtractor(PredictorInterface):

    def __init__(self, model_path, model_config=None):
        """
    init model

    Args:
      model_path: model file path
      model_config: config string for model to init, in json format
    """
        self.predictor = Predictor(model_path)
        self.output_name = 'feature'

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
            outputs = self.predictor.predict_batch(
                input_data, mode='extract')['neck'].data.cpu().numpy()
            for idx in range(len(image_tensor_list)):
                single_result = {}
                out = np.squeeze(outputs[idx])
                single_result[self.output_name] = out
                outputs_list.append(single_result)

        return outputs_list


@PREDICTORS.register_module()
class TorchFaceFeatureExtractor(PredictorInterface):

    def __init__(self, model_path, model_config=None):
        """
    init model, add a facedetect and align for img input.

    Args:
      model_path: model file path
      model_config: config string for model to init, in json format
    """

        # Forward compatibility, to support both pth(only face feature model) or tar.gz(face feature model + mtcnn model)
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            self.predictor = Predictor(model_path)
            self.detector = FaceDetector()
        else:
            face_model = glob('%s/*.pth' % model_path) + glob(
                '%s/*.pt' % model_path)
            assert (len(face_model) == 1)
            self.predictor = Predictor(face_model[0])

            mtcnn_weights = glob('%s/weights/*.npy' % model_path)
            if len(mtcnn_weights) != 3:
                print(
                    "User provide model_path doesn't contain mtcnn models, we try to load weights from http, might failed!"
                )
            self.detector = FaceDetector(dir_path=model_path)

        self.output_name = 'feature'

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

    def predict(self, input_data_list, batch_size=-1, detect_and_align=True):
        """
    using session run predict a number of samples using batch_size

    Args:
      input_data_list:  a list of numpy array or PIL.Image, each array is a sample to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
      detect_and_align: True to detect and align before feature extractor
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
        if batch_size > 0:
            num_batches = int(math.ceil(float(num_image) / batch_size))
            image_list = input_data_list
        else:
            num_batches = 1
            batch_size = len(input_data_list)
            image_list = input_data_list

        for i in range(len(image_list)):
            if isinstance(image_list[i], np.ndarray):
                image_list[i] = Image.fromarray(image_list[i])

        outputs_list = []
        for batch_idx in range(num_batches):
            batch_image_list = image_list[batch_idx * batch_size:min(
                (batch_idx + 1) * batch_size, len(image_list))]
            if detect_and_align:
                for idx, img in enumerate(batch_image_list):
                    bbox, ld = self.detector.safe_detect(img)
                    if len(bbox) > 1:
                        print('batch %d , %dth image has more then 1 face' %
                              (batch_idx, idx))
                        batch_image_list[idx] = np.array(
                            glint360k_align(img, ld[0]))
                    elif len(bbox) == 0:
                        print(
                            'batch %d , %dth image has no face detected, use original img'
                            % (batch_idx, idx))
                        batch_image_list[idx] = np.array(
                            img.resize((112, 112)))
                    else:
                        batch_image_list[idx] = np.array(
                            glint360k_align(img, ld[0]))
            else:
                for idx, img in enumerate(batch_image_list):
                    batch_image_list[idx] = np.array(img.resize((112, 112)))

            image_tensor_list = self.predictor.preprocess(batch_image_list)
            input_data = self.batch(image_tensor_list)
            outputs = self.predictor.predict_batch(
                input_data, mode='extract')['neck'].data.cpu().numpy()

            for idx in range(len(image_tensor_list)):
                single_result = {}
                out = np.squeeze(outputs[idx])
                single_result[self.output_name] = out
                outputs_list.append(single_result)

        return outputs_list


@PREDICTORS.register_module()
class TorchMultiFaceFeatureExtractor(PredictorInterface):

    def __init__(self, model_path, model_config=None):
        """
    init model, add a facedetect and align for img input.

    Args:
      model_path: model file path
      model_config: config string for model to init, in json format
    """

        # Forward compatibility, to support both pth(only face feature model) or tar.gz(face feature model + mtcnn model)
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            self.predictor = Predictor(model_path)
            self.detector = FaceDetector()
        else:
            face_model = glob('%s/*.pth' % model_path) + glob(
                '%s/*.pt' % model_path)
            assert (len(face_model) == 1)
            self.predictor = Predictor(face_model[0])

            mtcnn_weights = glob('%s/weights/*.npy' % model_path)
            if len(mtcnn_weights) != 3:
                print(
                    "User provide model_path doesn't contain mtcnn models, we try to load weights from http, might failed!"
                )
            self.detector = FaceDetector(dir_path=model_path)

        self.output_name = 'feature'

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

    def predict(self, input_data_list, batch_size=-1, detect_and_align=True):
        """
    using session run predict a number of samples using batch_size

    Args:
      input_data_list:  a list of numpy array or PIL.Image, each array is a sample to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
      detect_and_align: True to detect and align before feature extractor
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    Raise:
       if detect !=1 face in a img, then do nothing for this image
    """
        num_image = len(input_data_list)

        outputs_list = []
        for idx in range(num_image):
            img = input_data_list[idx]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            sub_imgs = []
            if detect_and_align:
                bboxes, ld = self.detector.safe_detect(img)
                if len(bboxes) > 1:
                    print('%dth image has more then 1 face' % idx)
                    for box, subld in zip(bboxes, ld):
                        sub_imgs.append(np.array(glint360k_align(img, subld)))
                    # sub_imgs.append(np.array(glint360k_align(img, ld[0])))
                elif len(bboxes) == 0:
                    print('%dth image has no face detected, use original img' %
                          idx)
                    sub_imgs.append(np.array(img.resize((112, 112))))
                else:
                    sub_imgs.append(np.array(glint360k_align(img, ld[0])))
            else:
                sub_imgs.append(np.array(img.resize((112, 112))))
                # x1,y1 x2,y2,score
                bboxes = [[0, 0, 111, 111, 1.0]]

            image_tensor_list = self.predictor.preprocess(sub_imgs)
            input_data = self.batch(image_tensor_list)
            outputs = self.predictor.predict_batch(
                input_data, mode='extract')['neck'].data.cpu().numpy()

            # for sub_idx in range(len(image_tensor_list)):
            single_result = {}
            if len(bboxes) == 0:
                bboxes = [[0, 0, 111, 111, 1.0]]

            # out = np.squeeze(outputs[0])

            single_result[self.output_name] = outputs
            single_result['bbox'] = bboxes

            outputs_list.append(single_result)

        return outputs_list


@PREDICTORS.register_module()
class TorchFaceAttrExtractor(PredictorInterface):

    def __init__(
        self,
        model_path,
        model_config=None,
        face_threshold=0.95,
        attr_method=['distribute_sum', 'softmax', 'softmax'],
        attr_name=['age', 'gender', 'emo'],
    ):
        """
    init model

    Args:
      model_path: model file path
      model_config: config string for model to init, in json format
      attr_method:
        - softmax: do softmax for feature_dim 1
        - distribute_sum: do softmax and prob sum
    """
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            self.predictor = Predictor(model_path)
            self.detector = FaceDetector()
        else:
            face_model = glob('%s/*.pth' % model_path) + glob(
                '%s/*.pt' % model_path)
            assert (len(face_model) == 1)
            self.predictor = Predictor(face_model[0])

            mtcnn_weights = glob('%s/weights/*.npy' % model_path)
            if len(mtcnn_weights) != 3:
                print(
                    "User provide model_path doesn't contain mtcnn models, we try to load weights from http, might failed!"
                )
            self.detector = FaceDetector(dir_path=model_path)

        self.attr_name = attr_name
        self.attr_method = attr_method
        assert (len(self.attr_method) == len(self.attr_name))
        self.gender_map = {0: 'female', 1: 'male'}
        self.emo_map = {
            0: 'Neutral',
            1: 'Happiness',
            2: 'Sadness',
            3: 'Surprise',
            4: 'Fear',
            5: 'Disgust',
            6: 'Anger',
            7: 'Contempt',
        }
        self.pop_map = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
        }
        self.face_threshold = face_threshold

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
            batch_image_list = image_list[
                batch_idx *
                batch_size:min(len(image_list), (batch_idx + 1) * batch_size)]

            face_image_list = []
            face_bbox_list = []
            faceidx_by_imageidx = {}

            for idx, img in enumerate(batch_image_list):
                # this try except only happens to no face detected
                bbox, ld = self.detector.safe_detect(img)

                if len(bbox) == 0:
                    print('batch %d , %dth image has no face detected' %
                          (batch_idx, idx))
                elif len(bbox) >= 1:
                    if len(bbox) > 1:
                        print('batch %d , %dth image has more then %d face' %
                              (batch_idx, idx, len(bbox)))

                    _bbox = []
                    _ld = []
                    for idx, b in enumerate(bbox):
                        if b[-1] > self.face_threshold:
                            _bbox.append(b)
                            _ld.append(ld[idx])

                    bbox = _bbox
                    ld = _ld

                    # this is for muti face detectd in one img
                    faceidx_by_imageidx[idx] = []
                    for bbox_idx, face_box in enumerate(bbox):
                        face_image_list.append(
                            glint360k_align(img, ld[bbox_idx]))
                        face_bbox_list.append(face_box)
                        face_idx = len(face_image_list) - 1
                        faceidx_by_imageidx[idx].append(face_idx)
                # else:
                #     batch_image_list[idx] = np.array(glint360k_align(img, ld[0]))

            if len(face_image_list) > 0:
                image_tensor_list = self.predictor.preprocess(face_image_list)
                input_data = self.batch(image_tensor_list)

                outputs = self.predictor.predict_batch(
                    input_data, mode='extract')

                neck_output_dict = {}
                for neck_idx, attr_method in enumerate(self.attr_method):
                    neck_output = outputs['neck_%d_0' % neck_idx]
                    neck_output = torch.nn.Softmax(dim=1)(neck_output)
                    if attr_method == 'softmax':
                        neck_output = torch.argmax(neck_output, dim=1)
                    elif attr_method == 'distribute_sum':
                        n, c = neck_output.size()
                        distribute = torch.arange(0, c).repeat(n, 1).to(
                            neck_output.device)
                        neck_output = (distribute * neck_output).sum(dim=1)
                    else:
                        raise Exception(
                            'TorchFaceAttrExtractor for neck %d only support attr_method softmax/distributed sum'
                            % (neck_idx))
                        neck_output = torch.argmax(neck_output, dim=1)
                    neck_output_dict[neck_idx] = neck_output.cpu().numpy()

                for imgidx in faceidx_by_imageidx.keys():
                    single_result = {}
                    for k in neck_output_dict.keys():
                        single_result['face_' + self.attr_name[k]] = []
                    single_result['face_bbox'] = []
                    for fn, faceidx in enumerate(faceidx_by_imageidx[imgidx]):
                        for k in neck_output_dict.keys():
                            out = np.squeeze(neck_output_dict[k][faceidx])
                            if self.attr_method[k] == 'softmax':
                                label_map = getattr(
                                    self, '%s_map' % self.attr_name[k])
                                out = label_map[out]
                            single_result['face_' +
                                          self.attr_name[k]].append(out)
                        single_result['face_bbox'].append(
                            face_bbox_list[faceidx])
                    outputs_list.append(single_result)
        return outputs_list
