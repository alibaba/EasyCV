# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import os.path as osp
import re
import tempfile
from argparse import ArgumentParser

import mmcv

from easycv.predictors import ClassificationPredictor, DetectionPredictor
from easycv.thirdparty.mot.bytetrack.byte_tracker import BYTETracker
from easycv.thirdparty.mot.utils import (
    _is_valid_video, detection_result_filter, gen_res,
    get_mtmct_matching_results, reid_predictor, save_mtmct_crops,
    save_mtmct_vis_results, show_result, sub_cluster, trajectory_fusion,
    video2frames)


def main():
    parser = ArgumentParser()
    parser.add_argument('--det_config', default=None, help='detection config file')
    parser.add_argument('--det_checkpoint', help='detection checkpoint file')
    parser.add_argument('--reid_config', default=None, help='reid config file')
    parser.add_argument('--reid_checkpoint', help='reid checkpoint file')
    parser.add_argument('--input', help='input video file folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument(
        '--save_images',
        action='store_true',
        help='Save visualization image results.')
    args = parser.parse_args()
    assert args.output

    cid_bias = dict()
    mot_list_breaks = []
    cid_tid_dict = dict()
    # load images
    seqs = os.listdir(args.input)
    for seq in sorted(seqs):
        fpath = os.path.join(args.input, seq)
        if os.path.isfile(fpath) and _is_valid_video(fpath):
            seq = seq.split('.')[-2]
            print('start tracking seq: {}'.format(seq))
            seq_dir = os.path.join(args.input, seq)
            if not os.path.exists(seq_dir):
                print('ffmpeg processing of video {}'.format(fpath))
                frames_path = video2frames(
                    video_path=fpath, outpath=args.input, frame_rate=25)
            else:
                print('ffmpeg has been successfully run. The {} directory already exists'.format(seq_dir))

            fpath = seq_dir
            if os.path.isdir(fpath) == False:
                print('{} is not a image folder.'.format(fpath))
                continue
            assert os.path.isdir(fpath), '{} should be a directory'.format(
                fpath)
            imgs = glob.glob(os.path.join(fpath, '*.jpg'))
            imgs.sort()
            assert len(imgs) > 0, '{} has no images.'.format(fpath)
        else:
            continue

        # build the model from a config file and a checkpoint file
        det_model = DetectionPredictor(
            args.det_checkpoint, args.det_config, score_threshold=0)
        reid_model = ClassificationPredictor(args.reid_checkpoint,
                                             args.reid_config)
        tracker = BYTETracker(
            det_high_thresh=0.2,
            det_low_thresh=0.05,
            match_thresh=1.0,
            match_thresh_second=1.0,
            match_thresh_init=1.0,
            track_buffer=2,
            frame_rate=25)

        # test and show/save the images
        track_result = None
        mot_features_dict = dict()
        for frame_id, img in enumerate(imgs):

            result = det_model(img)[0]

            detection_boxes = result['detection_boxes']
            detection_scores = result['detection_scores']
            detection_classes = result['detection_classes']
            img_metas = result['img_metas']

            detection_boxes, detection_scores, detection_classes = detection_result_filter(
                detection_boxes,
                detection_scores,
                detection_classes,
                target_classes=[2], # 0: person 2: car
                target_thresholds=[0])
            if len(detection_boxes) > 0:
                track_result = tracker.update(
                    detection_boxes, detection_scores,
                    detection_classes)  # [id, t, l, b, r, score]

                pred_embeddings, track_bboxes = reid_predictor(
                    {
                        'boxes': track_result['track_bboxes'],
                        'img_metas': img_metas
                    }, reid_model)

                for idx in range(len(track_bboxes)):
                    _id = int(track_bboxes[idx, 0])
                    imgname = f'{seq}_{_id}_{frame_id}.jpg'

                    mot_features_dict[imgname] = dict()
                    mot_features_dict[imgname]['bbox'] = track_bboxes[idx, 1:5]
                    mot_features_dict[imgname]['frame'] = f'{frame_id:06d}'
                    mot_features_dict[imgname]['id'] = _id
                    mot_features_dict[imgname]['imgname'] = imgname
                    mot_features_dict[imgname]['feat'] = pred_embeddings[idx].squeeze().numpy()

        cid = int(re.sub('[a-z,A-Z]', '', seq))
        cid_bias[cid] = float(0)
        tid_data, mot_list_break = trajectory_fusion(mot_features_dict, cid,
                                                     cid_bias)
        mot_list_breaks.append(mot_list_break)
        # single seq process
        for line in tid_data:
            tracklet = tid_data[line]
            tid = tracklet['tid']
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    scene_cluster = list(cid_bias.keys())

    map_tid = sub_cluster(
        cid_tid_dict,
        scene_cluster,
        use_ff=False,
        use_rerank=False,
        use_st_filter=False)

    pred_mtmct_file = os.path.join(args.output, 'mtmct_result.txt')
    gen_res(pred_mtmct_file, scene_cluster, map_tid, mot_list_breaks)

    camera_results, cid_tid_fid_res = get_mtmct_matching_results(
        pred_mtmct_file, secs_interval=0.5, video_fps=20)

    crops_dir = os.path.join(args.output, 'mtmct_crops')
    save_mtmct_crops(
        cid_tid_fid_res, images_dir=args.input, crops_dir=crops_dir)

    save_dir = os.path.join(args.output, 'mtmct_vis')
    save_mtmct_vis_results(
        camera_results,
        images_dir=args.input,
        save_dir=save_dir,
        save_videos=args.save_images)


if __name__ == '__main__':
    main()