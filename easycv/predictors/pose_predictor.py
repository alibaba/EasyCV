# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import json

import mmcv
import numpy as np
import torch
from mmcv.image import imwrite
from mmcv.utils.path import is_filepath
from mmcv.visualization.image import imshow

from easycv.core.visualization import imshow_bboxes, imshow_keypoints
from easycv.datasets.pose.data_sources.top_down import DatasetInfo
from easycv.datasets.pose.pipelines.transforms import bbox_cs2xyxy
from easycv.file import io
from easycv.predictors.builder import PREDICTORS, build_predictor
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.misc import deprecated
from .base import InputProcessor, OutputProcessor, PredictorV2

np.set_printoptions(suppress=True)


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


def vis_pose_result(
    model,
    img,
    result,
    radius=4,
    thickness=1,
    kpt_score_thr=0.3,
    bbox_color='green',
    dataset_info=None,
    out_file=None,
    pose_kpt_color=None,
    pose_link_color=None,
    text_color='white',
    font_scale=0.5,
    bbox_thickness=1,
    win_name='',
    show=False,
    wait_time=0,
):
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
        out_file (str|None): The filename of the output visualization image.
        show (bool): Whether to show the image. Default: False.
        wait_time (int): Value of waitKey param.
            Default: 0.
        out_file (str or None): The filename to write the image.
            Default: None.
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

    img = mmcv.imread(img)
    img = img.copy()

    bbox_result = result.get('bbox', [])
    pose_result = result['keypoints']

    if len(bbox_result) > 0:
        bboxes = np.vstack(bbox_result)
        labels = None
        if 'label' in result:
            labels = result['label']
        # draw bounding boxes
        imshow_bboxes(
            img,
            bboxes,
            labels=labels,
            colors=bbox_color,
            text_color=text_color,
            thickness=bbox_thickness,
            font_scale=font_scale,
            show=False)

    imshow_keypoints(img, pose_result, skeleton, kpt_score_thr, pose_kpt_color,
                     pose_link_color, radius, thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)

    return img


class PoseTopDownInputProcessor(InputProcessor):

    def __init__(self,
                 cfg,
                 dataset_info,
                 detection_predictor_config,
                 bbox_thr=None,
                 pipelines=None,
                 batch_size=1,
                 cat_id=None,
                 mode='BGR'):
        self.detection_predictor = build_predictor(detection_predictor_config)
        self.dataset_info = dataset_info
        self.bbox_thr = bbox_thr
        self.cat_id = cat_id
        super().__init__(
            cfg,
            pipelines=pipelines,
            batch_size=batch_size,
            threads=1,
            mode=mode)

    def get_detection_outputs(self, input, cat_id=None):
        det_results = self.detection_predictor(input['img'], keep_inputs=False)
        person_results = self._process_detection_results(
            det_results, cat_id=cat_id)
        return person_results

    def _process_detection_results(self, det_results, cat_id=None):
        """Process det results, and return a list of bboxes.

        Args:
            det_results (list|tuple): det results.
            cat_id (int | str): category id or name to reserve, if None, reserve all detection results.

        Returns:
            person_results (list): a list of detected bounding boxes
        """
        # Only support one sample/image
        if isinstance(det_results, tuple):
            det_results = det_results[0]
        elif isinstance(det_results, list):
            det_results = det_results[0]
        else:
            det_results = det_results

        bboxes = det_results['detection_boxes']
        scores = det_results['detection_scores']
        classes = det_results['detection_classes']

        if cat_id is not None:
            if isinstance(cat_id, str):
                assert cat_id in self.detection_predictor.cfg.CLASSES, f'cat_id "{cat_id}" not in detection classes list: {self.detection_predictor.cfg.CLASSES}'
                assert det_results.get('detection_class_names',
                                       None) is not None
                detection_class_names = det_results['detection_class_names']
                keeped_ids = [
                    i for i in range(len(detection_class_names))
                    if str(detection_class_names[i]) == str(cat_id)
                ]
            else:
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

    def process_single(self, input):
        output = super()._load_input(input)

        person_results = self.get_detection_outputs(output, cat_id=self.cat_id)

        box_id = 0

        # Select bboxes by score threshold
        bboxes = np.array([res['bbox'] for res in person_results])
        if self.bbox_thr is not None:
            assert bboxes.shape[1] == 5
            valid_idx = np.where(bboxes[:, 4] > self.bbox_thr)[0]
            bboxes = bboxes[valid_idx]
            person_results = [person_results[i] for i in valid_idx]

        results = []
        for person_result in person_results:
            box = person_result['bbox']  # x,y,x,y,s
            boxc = [box[0], box[1], box[2] - box[0],
                    box[3] - box[1]]  # x,y,w,h
            center, scale = _box2cs(self.cfg.data_cfg['image_size'], boxc)
            data = {
                'image_id':
                0,
                'center':
                center,
                'scale':
                scale,
                'bbox':
                box,
                'bbox_score':
                box[4] if len(box) == 5 else 1,
                'bbox_id':
                box_id,  # need to be assigned if batch_size > 1
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
                    'num_joints': self.cfg.data_cfg['num_joints'],
                },
                'image_file':
                output['filename'],
                'img':
                output['img'],
                'img_shape':
                output['img_shape'],
                'ori_shape':
                output['ori_shape'],
                'img_fields':
                output['img_fields'],
            }
            box_id += 1
            data_processor = self.processor(data)
            data_processor['bbox'] = box
            results.append(data_processor)

        return results

    def __call__(self, inputs):
        """Process all inputs list. And collate to batch and put to target device.
        If you need custom ops to load or process a batch samples, you need to reimplement it.
        """
        batch_outputs = []
        for inp in inputs:
            for res in self.process_single(inp):
                batch_outputs.append(res)

        if len(batch_outputs) < 1:
            return batch_outputs

        batch_outputs = self._collate_fn(batch_outputs)
        batch_outputs['img_metas']._data = [[
            img_meta[i] for img_meta in batch_outputs['img_metas']._data
            for i in range(len(img_meta))
        ]]
        return batch_outputs


class PoseTopDownOutputProcessor(OutputProcessor):

    def __call__(self, inputs):
        output = {}
        output['keypoints'] = inputs['preds']
        output['bbox'] = np.array(inputs['boxes'])  # x1, y1, x2, y2 score

        return output


# TODO: Fix when multi people are detected in each sample,
# all the people results will be passed to the pose model,
# resulting in a dynamic batch_size, which is not supported by jit script model.
@PREDICTORS.register_module()
class PoseTopDownPredictor(PredictorV2):
    """Pose topdown predictor.
        Args:
            model_path (str): Path of model path.
            config_file (Optinal[str]): Config file path for model and processor to init. Defaults to None.
            detection_model_config: Dict of person detection model predictor config,
                example like ``dict(type="", model_path="", config_file="", ......)``
            batch_size (int): Batch size for forward.
            bbox_thr (float): Bounding box threshold to filter output results of detection model
            cat_id (int | str): Category id or name to filter target objects.
            device (str | torch.device): Support str('cuda' or 'cpu') or torch.device, if is None, detect device automatically.
            save_results (bool): Whether to save predict results.
            save_path (str): File path for saving results, only valid when `save_results` is True.
            pipelines (list[dict]): Data pipeline configs.
            mode (str): The image mode into the model.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 detection_predictor_config=None,
                 batch_size=1,
                 bbox_thr=None,
                 cat_id=None,
                 device=None,
                 pipelines=None,
                 save_results=False,
                 save_path=None,
                 mode='BGR',
                 model_type=None,
                 *args,
                 **kwargs):
        assert batch_size == 1, 'Only support batch_size=1 now!'
        self.cat_id = cat_id
        self.bbox_thr = bbox_thr
        self.detection_predictor_config = detection_predictor_config

        self.model_type = model_type
        if self.model_type is None:
            if model_path.endswith('jit'):
                assert config_file is not None
                self.model_type = 'jit'
            elif model_path.endswith('blade'):
                import torch_blade
                assert config_file is not None
                self.model_type = 'blade'
            else:
                self.model_type = 'raw'
        assert self.model_type in ['raw', 'jit', 'blade']

        super(PoseTopDownPredictor, self).__init__(
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
        if hasattr(self.cfg, 'dataset_info'):
            dataset_info = self.cfg.dataset_info
            if is_filepath(dataset_info):
                cfg = mmcv_config_fromfile(dataset_info)
                dataset_info = cfg._cfg_dict['dataset_info']
        else:
            from easycv.datasets.pose.data_sources.coco import COCO_DATASET_INFO
            dataset_info = COCO_DATASET_INFO

        self.dataset_info = DatasetInfo(dataset_info)

    def _build_model(self):
        if self.model_type != 'raw':
            with io.open(self.model_path, 'rb') as infile:
                model = torch.jit.load(infile, self.device)
        else:
            model = super()._build_model()
        return model

    def prepare_model(self):
        """Build model from config file by default.
        If the model is not loaded from a configuration file, e.g. torch jit model, you need to reimplement it.
        """
        model = self._build_model()
        model.to(self.device)
        model.eval()
        if self.model_type == 'raw':
            load_checkpoint(model, self.model_path, map_location='cpu')
        return model

    def model_forward(self, inputs, return_heatmap=False):
        boxes = inputs['bbox'].cpu().numpy()
        if self.model_type == 'raw':
            with torch.no_grad():
                result = self.model(
                    **inputs, mode='test', return_heatmap=return_heatmap)
        else:
            img_metas = inputs['img_metas']
            with torch.no_grad():
                img = inputs['img'].to(self.device)
                tensor_img_metas = copy.deepcopy(img_metas)
                for meta in tensor_img_metas:
                    meta.pop('image_file')
                    for k, v in meta.items():
                        meta[k] = torch.tensor(v)
                output_heatmap = self.model(img, tensor_img_metas)

            from easycv.models.pose.heads.topdown_heatmap_base_head import decode_heatmap
            output_heatmap = output_heatmap.cpu().numpy()
            result = decode_heatmap(output_heatmap, img_metas,
                                    self.cfg.model.test_cfg)

        result['boxes'] = np.array(boxes)
        return result

    def get_input_processor(self):
        return PoseTopDownInputProcessor(
            cfg=self.cfg,
            dataset_info=self.dataset_info,
            detection_predictor_config=self.detection_predictor_config,
            bbox_thr=self.bbox_thr,
            pipelines=self.pipelines,
            batch_size=self.batch_size,
            cat_id=self.cat_id,
            mode=self.mode)

    def get_output_processor(self):
        return PoseTopDownOutputProcessor()

    def show_result(self,
                    image,
                    keypoints,
                    radius=4,
                    thickness=3,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    show=False,
                    save_path=None):
        vis_result = vis_pose_result(
            self.model,
            image,
            keypoints,
            dataset_info=self.dataset_info,
            radius=radius,
            thickness=thickness,
            kpt_score_thr=kpt_score_thr,
            bbox_color=bbox_color,
            show=show,
            out_file=save_path)

        return vis_result


class _TorchPoseTopDownOutputProcessor(PoseTopDownOutputProcessor):

    def __call__(self, inputs):
        output = super(_TorchPoseTopDownOutputProcessor, self).__call__(inputs)

        bbox = output['bbox']
        keypoints = output['keypoints']
        results = []
        for i in range(len(keypoints)):
            results.append({'bbox': bbox[i], 'keypoints': keypoints[i]})
        return {'pose_results': results}


@deprecated(reason='Please use PoseTopDownPredictor.')
@PREDICTORS.register_module()
class TorchPoseTopDownPredictorWithDetector(PoseTopDownPredictor):

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
        },
    ):
        """
        init model

        Args:
          model_path: pose and detection model file path, split with `,`,
                      make sure the first is pose model, second is detection model
          model_config: config string for model to init, in json format
        """
        if isinstance(model_config, str):
            model_config = json.loads(model_config)

        reserved_classes = model_config['detection'].pop(
            'reserved_classes', [])
        if len(reserved_classes) == 0:
            reserved_classes = None
        else:
            assert len(reserved_classes) == 1
            reserved_classes = reserved_classes[0]

        model_list = model_path.split(',')
        assert len(model_list) == 2
        # first is pose model, second is detection model
        pose_model_path, detection_model_path = model_list

        detection_model_type = model_config['detection'].pop('model_type')
        if detection_model_type == 'TorchYoloXPredictor':
            detection_predictor_config = dict(
                type=detection_model_type,
                model_path=detection_model_path,
                model_config=model_config['detection'])
        else:
            detection_predictor_config = dict(
                model_path=detection_model_path, **model_config['detection'])

        pose_kwargs = model_config['pose']
        pose_kwargs.pop('format', None)

        super().__init__(
            model_path=pose_model_path,
            detection_predictor_config=detection_predictor_config,
            cat_id=reserved_classes,
            **pose_kwargs,
        )

    def get_output_processor(self):
        return _TorchPoseTopDownOutputProcessor()

    def show_result(self,
                    image_path,
                    keypoints,
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    show=False,
                    save_path=None):
        dataset_info = self.dataset_info
        # get dataset info
        if (dataset_info is None and hasattr(self.model, 'cfg')
                and 'dataset_info' in self.model.cfg):
            dataset_info = DatasetInfo(self.model.cfg.dataset_info)

        if not dataset_info:
            raise ValueError('Please provide `dataset_info`!')

        skeleton = dataset_info.skeleton
        pose_kpt_color = dataset_info.pose_kpt_color
        pose_link_color = dataset_info.pose_link_color

        if hasattr(self.model, 'module'):
            self.model = self.model.module

        img = self.model.show_result(
            image_path,
            keypoints,
            skeleton,
            radius=radius,
            thickness=thickness,
            pose_kpt_color=pose_kpt_color,
            pose_link_color=pose_link_color,
            kpt_score_thr=kpt_score_thr,
            bbox_color=bbox_color,
            show=show,
            out_file=save_path)

        return img
