# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import cv2
import mmcv

from easycv.predictors import DetectionPredictor
from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker
from easycv.thirdparty.mot.utils import detection_result_filter, show_result
from .builder import PREDICTORS


@PREDICTORS.register_module()
class MOTPredictor(object):
    """MOT Predictor.


    Args:
        model_path (str): Path of model path.
        config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
        score_threshold(float): Specifies the filter score threshold for bbox.
        tracker_config (dict): Specify the parameters of the tracker.
        save_path (str): File path for saving results.
        fps: (int): Specify the fps of the output video.
    """

    def __init__(
            self,
            model_path,
            config_file=None,
            score_threshold=0.5,
            tracker_config={
                'det_high_thresh': 0.2,
                'det_low_thresh': 0.05,
                'match_thresh': 1.0,
                'match_thresh_second': 1.0,
                'match_thresh_init': 1.0,
                'track_buffer': 2,
                'frame_rate': 25
            },
            save_path=None,
            fps=24):

        self.model = DetectionPredictor(
            model_path, config_file, score_threshold=score_threshold)
        self.tracker = BYTETracker(**tracker_config)
        self.fps = fps
        self.output = save_path

    def __call__(self, inputs):
        # support list(dict(str)) as input
        if isinstance(inputs, str):
            inputs = [{'filename': inputs}]
        elif isinstance(inputs, list) and not isinstance(inputs[0], dict):
            tmp = []
            for input in inputs:
                tmp.append({'filename': input})
            inputs = tmp

        results = []
        for i in range(len(inputs)):
            # define input
            input = inputs[i]['filename']
            if osp.isdir(input):
                imgs = glob.glob(os.path.join(input, '*.jpg'))
                imgs.sort()
                IN_VIDEO = False
            else:
                imgs = mmcv.VideoReader(input)
                IN_VIDEO = True

            # define output
            if self.output is not None:
                if self.output.endswith('.mp4'):
                    OUT_VIDEO = True
                    out_dir = tempfile.TemporaryDirectory()
                    out_path = out_dir.name
                    _out = self.output.rsplit(os.sep, 1)
                    if len(_out) > 1:
                        os.makedirs(_out[0], exist_ok=True)
                else:
                    OUT_VIDEO = False
                    out_path = self.output
                    os.makedirs(out_path, exist_ok=True)

            prog_bar = mmcv.ProgressBar(len(imgs))

            # test and show/save the images
            track_result = None
            track_result_list = []
            for frame_id, img in enumerate(imgs):
                if osp.isdir(input):
                    timestamp = frame_id
                else:
                    seconds = imgs.vcap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    timestamp = seconds

                detection_results = self.model(img)[0]

                detection_boxes = detection_results['detection_boxes']
                detection_scores = detection_results['detection_scores']
                detection_classes = detection_results['detection_classes']

                detection_boxes, detection_scores, detection_classes = detection_result_filter(
                    detection_boxes,
                    detection_scores,
                    detection_classes,
                    target_classes=[0],
                    target_thresholds=[0])
                if len(detection_boxes) > 0:
                    track_result = self.tracker.update(
                        detection_boxes, detection_scores,
                        detection_classes)  # [id, t, l, b, r, score]
                    track_result['timestamp'] = timestamp
                    track_result_list.append(track_result)

                if self.output is not None:
                    if IN_VIDEO or OUT_VIDEO:
                        out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
                    else:
                        out_file = osp.join(out_path,
                                            img.rsplit(os.sep, 1)[-1])
                else:
                    out_file = None

                if out_file is not None:
                    show_result(
                        img,
                        track_result,
                        score_thr=0,
                        show=False,
                        wait_time=int(1000. / self.fps),
                        out_file=out_file)

                prog_bar.update()

            if self.output and OUT_VIDEO:
                print(
                    f'making the output video at {self.output} with a FPS of {self.fps}'
                )
                mmcv.frames2video(
                    out_path, self.output, fps=self.fps, fourcc='mp4v')
                out_dir.cleanup()

            results.append(track_result_list)

        return results
