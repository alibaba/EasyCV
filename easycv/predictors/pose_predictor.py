import functools
import json

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.utils.path import is_filepath
from torchvision.transforms import Compose

from easycv.datasets.pose.data_sources.top_down import DatasetInfo
from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.models import build_model
from easycv.predictors.builder import PREDICTORS
from easycv.predictors.detector import TorchYoloXPredictor
from easycv.utils.bbox_util import xywh2xyxy_coco, xyxy2xywh_coco
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.registry import build_from_cfg

try:
    from easy_vision.python.inference.predictor import PredictorInterface
except:
    from easycv.predictors.interface import PredictorInterface


class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the img_or_path.
                if `img_or_path` is str, return self.channel_order mode,
                if np.ndarray, return raw without process.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = mmcv.imread(results['img_or_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            img = results['img_or_path']
        else:
            raise TypeError(
                '"img_or_path" must be a numpy array or a str or a pathlib.Path object'
            )

        results['img'] = img
        return results


def _box2cs(image_size, box):
    """This encodes bbox(x,y,w,h) into (center, scale)
    Args:
        x, y, w, h
    Returns:
        tuple: A tuple containing center and scale.
        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    aspect_ratio = image_size[0] / image_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    scale = scale * 1.25

    return center, scale


def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class OutputHook:

    def __init__(self, module, outputs=None, as_tensor=False):
        self.outputs = outputs
        self.as_tensor = as_tensor
        self.layer_outputs = {}
        self.register(module)

    def register(self, module):

        def hook_wrapper(name):

            def hook(model, input, output):
                if self.as_tensor:
                    self.layer_outputs[name] = output
                else:
                    if isinstance(output, list):
                        self.layer_outputs[name] = [
                            out.detach().cpu().numpy() for out in output
                        ]
                    else:
                        self.layer_outputs[name] = output.detach().cpu().numpy(
                        )

            return hook

        self.handles = []
        if isinstance(self.outputs, (list, tuple)):
            for name in self.outputs:
                try:
                    layer = rgetattr(module, name)
                    h = layer.register_forward_hook(hook_wrapper(name))
                except ModuleNotFoundError as module_not_found:
                    raise ModuleNotFoundError(
                        f'Module {name} not found') from module_not_found
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class TorchPoseTopDownPredictor(PredictorInterface):
    """Inference a single image with a list of bounding boxes.
    """

    def __init__(self, model_path, model_config=None):
        """
        init model

        Args:
          model_path: model file path
          model_config: config string for model to init, in json format
        """
        bbox_thr = model_config.get('bbox_thr', 0.3)
        format = model_config.get('format', 'xywh')

        assert format in ['xyxy', 'xywh']

        self.model_path = model_path
        self.bbox_thr = bbox_thr

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        with io.open(self.model_path, 'rb') as infile:
            checkpoint = torch.load(infile, map_location='cpu')

        assert 'meta' in checkpoint and 'config' in checkpoint[
            'meta'], 'meta.config is missing from checkpoint'
        self.cfg = checkpoint['meta']['config']

        assert hasattr(self.cfg, 'dataset_info'), \
            'Not find dataset_info in checkpoint["meta"]["config"]'

        if is_filepath(self.cfg.dataset_info):
            cfg = mmcv_config_fromfile(self.cfg.dataset_info)
            self.cfg.dataset_info = cfg._cfg_dict['dataset_info']

        self.dataset_info = DatasetInfo(self.cfg.dataset_info)
        self.cfg.model.pretrained = None

        # build model
        self.model = build_model(self.cfg.model)

        map_location = 'cpu' if self.device == 'cpu' else 'cuda'
        self.ckpt = load_checkpoint(
            self.model, self.model_path, map_location=map_location)

        self.model.to(self.device)
        self.model.eval()

        # build pipeline
        channel_order = self.cfg.test_pipeline[0].get('channel_order', 'rgb')
        test_pipeline = [LoadImage(channel_order=channel_order)] + [
            build_from_cfg(p, PIPELINES) for p in self.cfg.test_pipeline
        ]
        self.test_pipeline = Compose(test_pipeline)

    def _inference_single_pose_model(self,
                                     model,
                                     img_or_path,
                                     bboxes,
                                     dataset_info=None,
                                     return_heatmap=False):
        """Inference human bounding boxes.

        num_bboxes: N
        num_keypoints: K

        Args:
            model (nn.Module): The loaded pose model.
            img_or_path (str | np.ndarray): Image filename or loaded image.
            bboxes (list | np.ndarray): All bounding boxes (with scores),
                shaped (N, 4) or (N, 5). (left, top, width, height, [score])
                where N is number of bounding boxes.
            dataset_info (DatasetInfo): A class containing all dataset info.
            outputs (list[str] | tuple[str]): Names of layers whose output is
                to be returned, default: None

        Returns:
            ndarray[NxKx3]: Predicted pose x, y, score.
            heatmap[N, K, H, W]: Model output heatmap.
        """

        cfg = self.cfg
        device = next(model.parameters()).device

        assert len(bboxes[0]) in [4, 5]

        dataset_name = getattr(dataset_info, 'dataset_name', '')
        flip_pairs = dataset_info.flip_pairs

        batch_data = []
        for bbox in bboxes:
            center, scale = _box2cs(cfg.data_cfg['image_size'], bbox)

            # prepare data
            data = {
                'img_or_path':
                img_or_path,
                'image_id':
                0,
                'center':
                center,
                'scale':
                scale,
                'bbox_score':
                bbox[4] if len(bbox) == 5 else 1,
                'bbox_id':
                0,  # need to be assigned if batch_size > 1
                'dataset':
                dataset_name,
                'joints_3d':
                np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'joints_3d_visible':
                np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'rotation':
                0,
                'ann_info': {
                    'image_size': np.array(cfg.data_cfg['image_size']),
                    'num_joints': cfg.data_cfg['num_joints'],
                    'flip_pairs': flip_pairs
                }
            }
            data = self.test_pipeline(data)
            batch_data.append(data)

        batch_data = collate(batch_data, samples_per_gpu=1)

        if next(model.parameters()).is_cuda:
            # scatter not work so just move image to cuda device
            batch_data['img'] = batch_data['img'].to(device)
        # get all img_metas of each bounding box
        batch_data['img_metas'] = [
            img_metas[0] for img_metas in batch_data['img_metas'].data
        ]

        # forward the model
        with torch.no_grad():
            result = model(
                img=batch_data['img'],
                mode='test',
                img_metas=batch_data['img_metas'],
                return_heatmap=return_heatmap)

        if return_heatmap:
            return result['preds'], result['output_heatmap']
        else:
            return result['preds'], None

    def _predict_single_img(self,
                            img_info,
                            bbox_thr,
                            dataset_info,
                            return_heatmap=False,
                            outputs=None):

        pose_results = []
        returned_outputs = []
        img_or_path = img_info['img']
        detection_results = img_info['detection_results']

        if not detection_results:
            return [], []

        # Change for-loop preprocess each bbox to preprocess all bboxes at once.
        bboxes = np.array([box['bbox'] for box in detection_results])

        # Select bboxes by score threshold
        if bbox_thr is not None:
            assert bboxes.shape[1] == 5
            valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
            bboxes = bboxes[valid_idx]
            detection_results = [detection_results[i] for i in valid_idx]

        if format == 'xyxy':
            bboxes_xyxy = bboxes
            bboxes_xywh = xyxy2xywh_coco(bboxes.copy(), 1)
        else:
            # format is already 'xywh'
            bboxes_xywh = bboxes
            bboxes_xyxy = xywh2xyxy_coco(bboxes.copy(), -1)

        # if bbox_thr remove all bounding box
        if len(bboxes_xywh) == 0:
            return [], []

        with OutputHook(self.model, outputs=outputs, as_tensor=False) as h:
            # poses is results['pred'] # N x 17x 3
            poses, heatmap = self._inference_single_pose_model(
                self.model,
                img_or_path,
                bboxes_xywh,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap)

            if return_heatmap:
                h.layer_outputs['heatmap'] = heatmap

            returned_outputs.append(h.layer_outputs)

        assert len(poses) == len(detection_results), print(
            len(poses), len(detection_results), len(bboxes_xyxy))
        for pose, detection_result, bbox_xyxy in zip(poses, detection_results,
                                                     bboxes_xyxy):
            pose_result = detection_result.copy()
            pose_result['keypoints'] = pose
            pose_result['bbox'] = bbox_xyxy
            pose_results.append(pose_result)

        return pose_results, returned_outputs

    def predict(self, input_data_list, batch_size=-1, return_heatmap=False):
        """Inference pose.

        Args:
            input_data_list: A list of image infos, like:
                [
                    {
                        'img' (str | np.ndarray, RGB):
                            Image filename or loaded image.
                        'detection_results'(list | np.ndarray):
                            All bounding boxes (with scores),
                            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
                            where N is number of bounding boxes.
                    },
                    ...
                ]
            batch_size: batch size
            return_heatmap: return heatmap value or not, default false.

        Returns:
            {
                'pose_results': list of ndarray[NxKx3]: Predicted pose x, y, score
                'pose_heatmap' (optional): list of heatmap[N, K, H, W]: Model output heatmap
            }


        """
        all_pose_results = []

        for img_info in input_data_list:
            pose_results, returned_outputs = \
                self._predict_single_img(img_info, self.bbox_thr, self.dataset_info)
            output = {'pose_results': pose_results}
            if return_heatmap:
                output.update({'pose_heatmap': returned_outputs})
            # must return dict to adapt to pai
            all_pose_results.append(output)

        return all_pose_results


@PREDICTORS.register_module()
class TorchPoseTopDownPredictorWithDetector(PredictorInterface):

    SUPPORT_DETECTION_PREDICTORS = {'TorchYoloXPredictor': TorchYoloXPredictor}

    def __init__(
        self,
        model_path,
        model_config={
            'pose': {
                'bbox_thr': 0.3,
                'format': 'xywh'
            },
            'detection': {
                'model_type': None,
                'reserved_classes': [],
                'score_thresh': 0.0,
            }
        }):
        """
        init model

        Args:
          model_path: pose and detection model file path, split with `,`,
                      make sure the first is pose model, second is detection model
          model_config: config string for model to init, in json format
        """
        if isinstance(model_config, str):
            model_config = json.loads(model_config)

        detection_model_type = model_config['detection'].pop('model_type')
        assert detection_model_type in self.SUPPORT_DETECTION_PREDICTORS

        self.reserved_classes = model_config['detection'].get(
            'reserved_classes', [])

        model_list = model_path.split(',')
        assert len(model_list) == 2
        # first is pose model, second is detection model
        pose_model_path, detection_model_path = model_list

        detection_obj = self.SUPPORT_DETECTION_PREDICTORS[detection_model_type]
        self.detection_predictor = detection_obj(
            detection_model_path, model_config=model_config['detection'])
        self.pose_predictor = TorchPoseTopDownPredictor(
            pose_model_path, model_config=model_config['pose'])

    def process_det_results(self,
                            outputs,
                            input_data_list,
                            reserved_classes=[]):
        filter_outputs = []
        assert len(outputs) == len(input_data_list)
        for reserved_class in reserved_classes:
            assert reserved_class in self.detection_predictor.CLASSES, \
                '%s not in detection classes %s' % (reserved_class, self.detection_predictor.CLASSES)

        # if reserved_class if [], reserve all classes
        reserved_classes = reserved_classes or self.detection_predictor.CLASSES

        for i in range(len(outputs)):
            output = outputs[i]
            cur_data = {'img': input_data_list[i], 'detection_results': []}
            for class_name in output['detection_class_names']:
                if class_name in reserved_classes:
                    cur_data['detection_results'].append({
                        'bbox':
                        np.append(output['detection_boxes'][i],
                                  output['detection_scores'][i])
                    })
            filter_outputs.append(cur_data)

        return filter_outputs

    def predict(self, input_data_list, batch_size=-1, return_heatmap=False):
        """Inference with pose model and detection model.

        Args:
            input_data_list: A list of images(np.ndarray, RGB)
            batch_size: batch size
            return_heatmap: return heatmap value or not, default false.

        Returns:
            {
                'pose_results': list of ndarray[NxKx3]: Predicted pose x, y, score
                'pose_heatmap' (optional): list of heatmap[N, K, H, W]: Model output heatmap
            }


        """
        detection_output = self.detection_predictor.predict(input_data_list)
        output = self.process_det_results(detection_output, input_data_list,
                                          self.reserved_classes)
        pose_output = self.pose_predictor.predict(
            output, return_heatmap=return_heatmap)

        return pose_output


def vis_pose_result(model,
                    img,
                    result,
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset_info=None,
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """

    # get dataset info
    if (dataset_info is None and hasattr(model, 'cfg')
            and 'dataset_info' in model.cfg):
        dataset_info = DatasetInfo(model.cfg.dataset_info)

    if not dataset_info:
        raise ValueError('Please provide `dataset_info`!')

    skeleton = dataset_info.skeleton
    pose_kpt_color = dataset_info.pose_kpt_color
    pose_link_color = dataset_info.pose_link_color

    if hasattr(model, 'module'):
        model = model.module

    img = model.show_result(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_link_color=pose_link_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img
