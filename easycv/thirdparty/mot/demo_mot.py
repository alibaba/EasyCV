# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv

from easycv.predictors import DetectionPredictor
from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker
from easycv.thirdparty.mot.utils import detection_result_filter, show_result


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    assert args.output or args.show
    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
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

    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # build the model from a config file and a checkpoint file
    model = DetectionPredictor(args.checkpoint, args.config, score_threshold=0)
    tracker = BYTETracker(
        det_high_thresh=0.2,
        det_low_thresh=0.05,
        match_thresh=1.0,
        match_thresh_second=1.0,
        match_thresh_init=1.0,
        track_buffer=2,
        frame_rate=25)

    prog_bar = mmcv.ProgressBar(len(imgs))

    # test and show/save the images
    track_result = None
    for idx, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(args.input, img)
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
                out_file = osp.join(out_path, f'{idx:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        # if len(track_result['track_bboxes']) > 0:
        show_result(
            img,
            track_result,
            score_thr=args.score_thr,
            show=args.show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()


if __name__ == '__main__':
    main()
