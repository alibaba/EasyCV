# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import cv2
import os
import os.path as osp
import tempfile
import numpy as np
from argparse import ArgumentParser
import random
import seaborn as sns

import mmcv

from easycv.predictors import DetectionPredictor
from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker

def detection_result_filter(bboxes, scores, classes, target_classes, target_thresholds=None):
    # post process to filter result
    bboxes_tmp = []
    scores_tmp = []
    classes_tmp = []
    assert len(target_classes)==len(target_thresholds), "detection post process, class filter need target_classes and target_thresholds both, and should be same length!"

    for bidx, bcls in enumerate(classes):
        if bcls in target_classes and scores[bidx] > target_thresholds[target_classes.index(bcls)]:
            bboxes_tmp.append(bboxes[bidx])
            scores_tmp.append(scores[bidx])
            classes_tmp.append(classes[bidx])
    bboxes = np.array(bboxes_tmp)
    scores = np.array(scores_tmp)
    classes = np.array(classes_tmp)
    return bboxes, scores, classes

def results2outs(bbox_results=None,
                 **kwargs):
    """Restore the results (list of results of each category) into the results
    of the model forward.

    Args:
        bbox_results (list[np.ndarray]): Each list denotes bboxes of one
            category.

    Returns:
        tuple: tracking results of each class. It may contain keys as belows:

        - bboxes (np.ndarray): shape (n, 5)
        - ids (np.ndarray): shape (n, )
    """
    outputs = dict()

    if len(bbox_results) > 0:

        bboxes = bbox_results
        if bboxes.shape[1] == 5:
            outputs['bboxes'] = bboxes
        elif bboxes.shape[1] == 6:
            ids = bboxes[:, 0].astype(np.int64)
            bboxes = bboxes[:, 1:]
            outputs['bboxes'] = bboxes
            outputs['ids'] = ids
        else:
            raise NotImplementedError(
                f'Not supported bbox shape: (N, {bboxes.shape[1]})')

    return outputs

def random_color(seed):
    """Random a color according to the input seed."""
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color

def imshow_tracks(img,
                bboxes,
                ids,
                classes=None,
                score_thr=0.0,
                thickness=2,
                font_scale=0.4,
                show=False,
                wait_time=0,
                out_file=None):
    """Show the tracks with opencv."""
    if isinstance(img, str):
        img = mmcv.imread(img)
    if bboxes is not None and ids is not None:
        assert bboxes.ndim == 2
        assert ids.ndim == 1
        assert bboxes.shape[1] == 5

        img_shape = img.shape
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

        inds = np.where(bboxes[:, -1] > score_thr)[0]
        bboxes = bboxes[inds]
        ids = ids[inds]

        text_width, text_height = 9, 13
        for i, (bbox, id) in enumerate(zip(bboxes, ids)):
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
            score = float(bbox[-1])

            # bbox
            bbox_color = random_color(id)
            bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

            # score
            text = '{:.02f}'.format(score)
            width = len(text) * text_width
            img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
            cv2.putText(
                img,
                text, (x1, y1 + text_height - 2),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                color=(0, 0, 0))

            # id
            text = str(id)
            width = len(text) * text_width
            img[y1 + text_height:y1 + 2 * text_height,
                x1:x1 + width, :] = bbox_color
            cv2.putText(
                img,
                str(id), (x1, y1 + 2 * text_height - 2),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                color=(0, 0, 0))

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img

def show_result(
                img,
                result,
                score_thr=0.0,
                thickness=1,
                font_scale=0.5,
                show=False,
                out_file=None,
                wait_time=0,
                **kwargs):
    """Visualize tracking results.

    Args:
        img (str | ndarray): Filename of loaded image.
        result (dict): Tracking result.
            - The value of key 'track_bboxes' is list with length
            num_classes, and each element in list is ndarray with
            shape(n, 6) in [id, tl_x, tl_y, br_x, br_y, score] format.
            - The value of key 'det_bboxes' is list with length
            num_classes, and each element in list is ndarray with
            shape(n, 5) in [tl_x, tl_y, br_x, br_y, score] format.
        thickness (int, optional): Thickness of lines. Defaults to 1.
        font_scale (float, optional): Font scales of texts. Defaults
            to 0.5.
        show (bool, optional): Whether show the visualizations on the
            fly. Defaults to False.
        out_file (str | None, optional): Output filename. Defaults to None.

    Returns:
        ndarray: Visualized image.
    """
    assert isinstance(result, dict)
    track_bboxes = result.get('track_bboxes', None)
    if isinstance(img, str):
        img = mmcv.imread(img)
    outs_track = results2outs(bbox_results=track_bboxes)
    img = imshow_tracks(
        img,
        outs_track.get('bboxes', None),
        outs_track.get('ids', None),
        score_thr=score_thr,
        thickness=thickness,
        font_scale=font_scale,
        show=show,
        out_file=out_file,
        wait_time=wait_time)
    return img

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

        detection_boxes, detection_scores, detection_classes = detection_result_filter(detection_boxes, detection_scores, detection_classes, target_classes=[0], target_thresholds=[0])
        if len(detection_boxes) > 0:
            track_result = tracker.update(detection_boxes, detection_scores, detection_classes) # [id, t, l, b, r, score]

        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{idx:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        # if len(track_result['track_bboxes']) > 0:
        show_result(img, track_result,
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
