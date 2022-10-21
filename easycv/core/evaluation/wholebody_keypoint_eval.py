# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from
# https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/base/kpt_2d_sview_rgb_img_top_down_dataset.py
import numpy as np
from xtcocotools.cocoeval import COCOeval

from .builder import EVALUATORS
from .coco_evaluation import CoCoPoseTopDownEvaluator
from .metric_registry import METRICS
from .top_down_eval import (keypoint_auc, keypoint_epe, keypoint_nme,
                            keypoint_pck_accuracy)


@EVALUATORS.register_module
class WholeBodyKeyPointEvaluator(CoCoPoseTopDownEvaluator):
    """ KeyPoint evaluator.
    """

    def __init__(self, dataset_name=None, metric_names=['AP']):
        """

        Args:
            dataset_name: eval dataset name
            metric_names: eval metrics name
        """
        super(WholeBodyKeyPointEvaluator,
              self).__init__(dataset_name, metric_names)
        self.metric = metric_names
        self.dataset_name = dataset_name

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

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

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)

        cuts = np.cumsum([
            0, self.body_num, self.foot_num, self.face_num, self.left_hand_num,
            self.right_hand_num
        ])

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_body',
            self.sigmas[cuts[0]:cuts[1]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_foot',
            self.sigmas[cuts[1]:cuts[2]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_face',
            self.sigmas[cuts[2]:cuts[3]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_lefthand',
            self.sigmas[cuts[3]:cuts[4]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_righthand',
            self.sigmas[cuts[4]:cuts[5]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_wholebody',
            self.sigmas,
            use_area=True)
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
