# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
import glob

import mmcv

from easycv.predictors import DetectionPredictor
from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker
from easycv.thirdparty.mot.utils import detection_result_filter, show_result


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    args = parser.parse_args()

    # load images
    if osp.isdir(args.input):
        imgs = glob.glob(os.path.join(args.input, '*.jpg'))
        imgs.sort()
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True

    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = DetectionPredictor(args.checkpoint, args.config, score_threshold=0.2) # detr-like score_threshold set to 0
    tracker = BYTETracker(
        det_high_thresh=0.2,
        det_low_thresh=0.05,
        match_thresh=1.0,
        match_thresh_second=1.0,
        match_thresh_init=1.0,
        track_buffer=2,
        frame_rate=25)

    fps = 24
    prog_bar = mmcv.ProgressBar(len(imgs))

    # test and show/save the images
    track_result = None
    for frame_id, img in enumerate(imgs):
        result = model(img)[0]

        detection_boxes = result['detection_boxes']
        detection_scores = result['detection_scores']
        detection_classes = result['detection_classes']

        detection_boxes, detection_scores, detection_classes = detection_result_filter(
            detection_boxes,
            detection_scores,
            detection_classes,
            target_classes=[0],
            target_thresholds=[0])
        if len(detection_boxes) > 0:
            track_result = tracker.update(
                detection_boxes, detection_scores,
                detection_classes)  # [id, t, l, b, r, score]

        if args.output is not None:
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
            wait_time=int(1000. / fps),
            out_file=out_file)

        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    main()
