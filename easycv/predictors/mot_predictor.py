# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import cv2
import mmcv

from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker
from easycv.thirdparty.mot.utils import detection_result_filter, show_result
from .builder import PREDICTORS, build_predictor


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
            model_path=None,
            config_file=None,
            detection_predictor_config={
                'type': 'DetectionPredictor',
                'model_path': None,
                'config_file': None,
                'score_threshold': 0.5
            },
            tracker_config={
                'det_high_thresh': 0.2,
                'det_low_thresh': 0.05,
                'match_thresh': 1.0,
                'match_thresh_second': 1.0,
                'match_thresh_init': 1.0,
                'track_buffer': 2,
                'frame_rate': 25
            },
            show_result_config={
                'score_thr': 0,
                'show': False
            },
            save_path=None,
            IN_VIDEO=False,
            OUT_VIDEO=False,
            out_dir=None,
            fps=24):

        if model_path is not None:
            detection_predictor_config['model_path'] = model_path
        if config_file is not None:
            detection_predictor_config['config_file'] = config_file
        self.model = build_predictor(detection_predictor_config)
        self.tracker = BYTETracker(**tracker_config)
        self.fps = fps
        self.show_result_config = show_result_config
        self.output = save_path
        self.IN_VIDEO = IN_VIDEO
        self.OUT_VIDEO = OUT_VIDEO
        self.out_dir = out_dir

    def define_input(self, inputs):
        # support list(dict(str)) as input
        if isinstance(inputs, str):
            inputs = [{'filename': inputs}]
        elif isinstance(inputs, list) and not isinstance(inputs[0], dict):
            tmp = []
            for input in inputs:
                tmp.append({'filename': input})
            inputs = tmp

        # define input
        input = inputs[0]['filename']
        if osp.isdir(input):
            imgs = glob.glob(os.path.join(input, '*.jpg'))
            imgs.sort()
            self.IN_VIDEO = False
        else:
            imgs = mmcv.VideoReader(input)
            self.IN_VIDEO = True

        return imgs, input

    def define_output(self):
        if self.output is not None:
            if self.output.endswith('.mp4'):
                self.OUT_VIDEO = True
                self.out_dir = tempfile.TemporaryDirectory()
                out_path = self.out_dir.name
                _out = self.output.rsplit(os.sep, 1)
                if len(_out) > 1:
                    os.makedirs(_out[0], exist_ok=True)
            else:
                self.OUT_VIDEO = False
                out_path = self.output
                os.makedirs(out_path, exist_ok=True)
        else:
            out_path = None
        return out_path

    def __call__(self, inputs):
        # define input
        imgs, input = self.define_input(inputs)
        # define output
        out_path = self.define_output()

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
                if self.IN_VIDEO or self.OUT_VIDEO:
                    out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
            else:
                out_file = None

            if out_file is not None:
                show_result(
                    img,
                    track_result,
                    wait_time=int(1000. / self.fps),
                    out_file=out_file,
                    **self.show_result_config)
            prog_bar.update()

        if self.output and self.OUT_VIDEO:
            print(
                f'making the output video at {self.output} with a FPS of {self.fps}'
            )
            mmcv.frames2video(
                out_path, self.output, fps=self.fps, fourcc='mp4v')
            self.out_dir.cleanup()

        return [track_result_list]
