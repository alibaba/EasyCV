# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import tempfile
import re
import glob
from argparse import ArgumentParser

import mmcv

from easycv.predictors import DetectionPredictor, ClassificationPredictor
from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker
from easycv.thirdparty.mot.utils import detection_result_filter, show_result, reid_predictor, parse_bias, trajectory_fusion, video2frames, _is_valid_video, sub_cluster

def main():
    parser = ArgumentParser()
    parser.add_argument('--det_config', help='detection config file')
    parser.add_argument('--det_checkpoint', help='detection checkpoint file')
    parser.add_argument('--reid_config', help='reid config file')
    parser.add_argument('--reid_checkpoint', help='reid checkpoint file')
    parser.add_argument('--input', help='input video file folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
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

    cid_bias = parse_bias({'c003': 0, 'c004': 0})
    scene_cluster = list(cid_bias.keys())
    mot_list_breaks = []
    cid_tid_dict = dict()
    # load images
    seqs = os.listdir(args.input)
    for seq in sorted(seqs):
        fpath = os.path.join(args.input, seq)
        if os.path.isfile(fpath) and _is_valid_video(fpath):
            seq = seq.split('.')[-2]
            print('ffmpeg processing of video {}'.format(fpath))
            frames_path = video2frames(
                video_path=fpath, outpath=args.input, frame_rate=25)
            fpath = os.path.join(args.input, seq)

            if os.path.isdir(fpath) == False:
                print('{} is not a image folder.'.format(fpath))
                continue
            if os.path.exists(os.path.join(fpath, 'img1')):
                fpath = os.path.join(fpath, 'img1')
            assert os.path.isdir(fpath), '{} should be a directory'.format(
                fpath)
            imgs = glob.glob(os.path.join(fpath, '*.jpg'))
            imgs.sort()
            assert len(imgs) > 0, '{} has no images.'.format(fpath)
            print('start tracking seq: {}'.format(seq))
            
        IN_VIDEO = False

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
        det_model = DetectionPredictor(args.det_checkpoint, args.det_config, score_threshold=0)
        reid_model = ClassificationPredictor(args.reid_checkpoint, args.reid_config)
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
        mot_features_dict = dict()
        for frame_id, img in enumerate(imgs):
            if frame_id == 5:
                break
            result = det_model(img)[0]

            detection_boxes = result['detection_boxes']
            detection_scores = result['detection_scores']
            detection_classes = result['detection_classes']
            img_metas = result['img_metas']

            detection_boxes, detection_scores, detection_classes = detection_result_filter(detection_boxes, detection_scores, detection_classes, target_classes=[2], target_thresholds=[0])
            if len(detection_boxes) > 0:
                track_result = tracker.update(detection_boxes, detection_scores, detection_classes) # [id, t, l, b, r, score]

                pred_embeddings, track_bboxes  = reid_predictor({'boxes': track_result['track_bboxes'], 'img_metas': img_metas}, reid_model)


                for idx in range(len(track_bboxes)):
                    _id = int(track_bboxes[idx, 0])
                    imgname = f'{seq}_{_id}_{frame_id}.jpg'

                    mot_features_dict[imgname] = dict()
                    mot_features_dict[imgname]['bbox'] = track_bboxes[idx, 1:5]
                    mot_features_dict[imgname]['frame'] = f"{frame_id:06d}"
                    mot_features_dict[imgname]['id'] = _id
                    mot_features_dict[imgname]['imgname'] = imgname
                    mot_features_dict[imgname]['feat'] = pred_embeddings[idx]['prob'].squeeze().numpy()

        cid = int(re.sub('[a-z,A-Z]', "", seq))
        tid_data, mot_list_break = trajectory_fusion(
            mot_features_dict,
            cid,
            cid_bias)
        mot_list_breaks.append(mot_list_break)
        # single seq process
        for line in tid_data:
            tracklet = tid_data[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    map_tid = sub_cluster(
        cid_tid_dict,
        scene_cluster,
        use_ff=False,
        use_rerank=False,
        use_st_filter=False)
    print(map_tid)
    exit()

    if args.output is not None:
        if IN_VIDEO or OUT_VIDEO:
            out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
        else:
            out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
    else:
        out_file = None

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
