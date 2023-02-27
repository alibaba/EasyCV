# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import cv2

from easycv.predictors import DetectionPredictor
from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker
from easycv.thirdparty.mot.utils import detection_result_filter, show_result
from .base import PredictorV2
from .builder import PREDICTORS


@PREDICTORS.register_module()
class MOTPredictor(PredictorV2):
    """Generic MOT Predictor, it will filter bbox results by ``score_threshold`` .

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
                 fps=24,
                 *arg,
                 **kwargs):
        super(MOTPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=input_processor_threads,
            mode=mode)

        self.model = DetectionPredictor(
            model_path, config_file, score_threshold=score_threshold)
        self.tracker = BYTETracker(
            det_high_thresh=0.2,
            det_low_thresh=0.05,
            match_thresh=1.0,
            match_thresh_second=1.0,
            match_thresh_init=1.0,
            track_buffer=2,
            frame_rate=25)
        self.fps = fps

    def __call__(self, input, output):
        # define input
        if osp.isdir(input):
            imgs = glob.glob(os.path.join(input, '*.jpg'))
            imgs.sort()
            IN_VIDEO = False
        else:
            imgs = mmcv.VideoReader(input)
            IN_VIDEO = True

        # define output
        if output is not None:
            if output.endswith('.mp4'):
                OUT_VIDEO = True
                out_dir = tempfile.TemporaryDirectory()
                out_path = out_dir.name
                _out = output.rsplit(os.sep, 1)
                if len(_out) > 1:
                    os.makedirs(_out[0], exist_ok=True)
            else:
                OUT_VIDEO = False
                out_path = output
                os.makedirs(out_path, exist_ok=True)

        prog_bar = mmcv.ProgressBar(len(imgs))

        # test and show/save the images
        track_result_list = []
        for frame_id, img in enumerate(imgs):
            if osp.isdir(input):
                timestamp = frame_id
            else:
                timestamp = imgs.vcap.get(cv2.CAP_PROP_POS_MSEC)

            result = self.model(img)[0]

            detection_boxes = result['detection_boxes']
            detection_scores = result['detection_scores']
            detection_classes = result['detection_classes']

            detection_boxes, detection_scores, detection_classes = detection_result_filter(
                detection_boxes,
                detection_scores,
                detection_classes,
                target_classes=[2],
                target_thresholds=[0])
            if len(detection_boxes) > 0:
                track_result = self.tracker.update(
                    detection_boxes, detection_scores,
                    detection_classes)  # [id, t, l, b, r, score]
                track_result_list.append(track_result.insert(0, timestamp))

            if output is not None:
                if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
                else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
            else:
                out_file = None

            show_result(
                img,
                track_result,
                score_thr=0,
                show=False,
                wait_time=int(1000. / self.fps),
                out_file=out_file)

            prog_bar.update()

        if output and OUT_VIDEO:
            print(f'making the output video at {output} with a FPS of {self.fps}')
            mmcv.frames2video(out_path, output, fps=self.fps, fourcc='mp4v')
            out_dir.cleanup()
        
        return track_result_list
