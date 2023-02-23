# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from
# https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/base/kpt_2d_sview_rgb_img_top_down_dataset.py
import json
import os
import tempfile

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from easycv.utils.json_utils import MyEncoder
from .builder import EVALUATORS
from .coco_evaluation import CoCoPoseTopDownEvaluator
from .metric_registry import METRICS


@EVALUATORS.register_module
class WholeBodyKeyPointEvaluator(CoCoPoseTopDownEvaluator):
    """ KeyPoint evaluator.
    """

    def __init__(self, dataset_name=None, metric_names=['AP'], **kwargs):
        """

        Args:
            dataset_name: eval dataset name
            metric_names: eval metrics name
        """
        super(WholeBodyKeyPointEvaluator,
              self).__init__(dataset_name, metric_names, **kwargs)
        self.metric = metric_names
        self.dataset_name = dataset_name
        self.body_num = kwargs.get('body_num', 17)
        self.foot_num = kwargs.get('foot_num', 6)
        self.face_num = kwargs.get('face_num', 68)
        self.left_hand_num = kwargs.get('left_hand_num', 21)
        self.right_hand_num = kwargs.get('right_hand_num', 21)

    def _coco_keypoint_results_one_category_kernel(self,
                                                   data_pack,
                                                   num_joints=None):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1, num_joints * 3)

            cuts = np.cumsum([
                0, self.body_num, self.foot_num, self.face_num,
                self.left_hand_num, self.right_hand_num
            ]) * 3

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point[cuts[0]:cuts[1]].tolist(),
                'foot_kpts': key_point[cuts[1]:cuts[2]].tolist(),
                'face_kpts': key_point[cuts[2]:cuts[3]].tolist(),
                'lefthand_kpts': key_point[cuts[3]:cuts[4]].tolist(),
                'righthand_kpts': key_point[cuts[4]:cuts[5]].tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, results, groundtruth, sigmas=None):
        """Keypoint evaluation using COCOAPI."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            groundtruth_file = os.path.join(
                tmp_dir, 'groundtruth_wholebody_keypoints.json')
            with open(groundtruth_file, 'w') as f:
                json.dump(groundtruth, f, sort_keys=True, indent=4)
            coco = COCO(groundtruth_file)

            res_file = os.path.join(tmp_dir, 'result_wholebody_keypoints.json')
            with open(res_file, 'w') as f:
                json.dump(results, f, sort_keys=True, indent=4, cls=MyEncoder)
            coco_det = coco.loadRes(res_file)

        cuts = np.cumsum([
            0, self.body_num, self.foot_num, self.face_num, self.left_hand_num,
            self.right_hand_num
        ])

        coco_eval = COCOeval(
            coco,
            coco_det,
            'keypoints_body',
            sigmas[cuts[0]:cuts[1]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            coco,
            coco_det,
            'keypoints_foot',
            sigmas[cuts[1]:cuts[2]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            coco,
            coco_det,
            'keypoints_face',
            sigmas[cuts[2]:cuts[3]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            coco,
            coco_det,
            'keypoints_lefthand',
            sigmas[cuts[3]:cuts[4]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            coco,
            coco_det,
            'keypoints_righthand',
            sigmas[cuts[4]:cuts[5]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            coco, coco_det, 'keypoints_wholebody', sigmas, use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str


# METRICS.register_default_best_metric(KeyPointEvaluator, 'PCK', 'max')
METRICS.register_default_best_metric(WholeBodyKeyPointEvaluator, 'AP', 'max')
