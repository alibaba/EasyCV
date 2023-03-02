import argparse
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch

from easycv.file.utils import is_url_path
from easycv.predictors.pose_predictor import PoseTopDownPredictor
from easycv.predictors.video_classifier import STGCNPredictor

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1
TMP_DIR = './tmp'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Video classification demo based skeleton.')
    parser.add_argument(
        '--video',
        default=
        'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/demos/videos/ntu_sample.avi',
        help='video file/url')
    parser.add_argument(
        '--out_file',
        default=f'{TMP_DIR}/demo_show.mp4',
        help='output filename')
    parser.add_argument(
        '--config',
        default=(
            'configs/video_recognition/stgcn/stgcn_80e_ntu60_xsub_keypoint.py'
        ),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=
        ('http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/video/skeleton_based/stgcn/stgcn_80e_ntu60_xsub.pth'
         ),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='configs/detection/yolox/yolox_s_8xb16_300e_coco.py',
        help='human detection config file path')
    parser.add_argument(
        '--det-checkpoint',
        default=
        ('http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/yolox/yolox_s_bs16_lr002/epoch_300.pt'
         ),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-predictor-type',
        default='YoloXPredictor',
        help='detection predictor type')
    parser.add_argument(
        '--pose-config',
        default='configs/pose/hrnet_w48_coco_256x192_udp.py',
        help='human pose estimation config file path')
    parser.add_argument(
        '--pose-checkpoint',
        default=
        ('http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/pose/top_down_hrnet/pose_hrnet_epoch_210_export.pt'
         ),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.5,
        help='the threshold of human detection score')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args


def frame_extraction(video_path, short_side):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    if is_url_path(video_path):
        from torch.hub import download_url_to_file
        cache_video_path = os.path.join(TMP_DIR, os.path.basename(video_path))
        print(
            'Download video file from remote to local path "{cache_video_path}"...'
        )
        download_url_to_file(video_path, cache_video_path)
        video_path = cache_video_path

    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join(TMP_DIR, osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
        frame = mmcv.imresize(frame, (new_w, new_h))
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def main():
    args = parse_args()

    if not osp.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    frame_paths, original_frames = frame_extraction(args.video,
                                                    args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get Human detection results
    pose_predictor = PoseTopDownPredictor(
        model_path=args.pose_checkpoint,
        config_file=args.pose_config,
        detection_predictor_config=dict(
            type=args.det_predictor_type,
            model_path=args.det_checkpoint,
            config_file=args.det_config,
        ),
        bbox_thr=args.bbox_thr,
        cat_id=0,  # person category id
    )

    video_cls_predictor = STGCNPredictor(
        model_path=args.checkpoint,
        config_file=args.config,
        ori_image_size=(w, h),
        label_map=None)

    pose_results = pose_predictor(original_frames)

    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_person = max([len(x) for x in pose_results])

    num_keypoint = 17
    keypoints = np.zeros((num_person, num_frame, num_keypoint, 2),
                         dtype=np.float16)
    keypoints_score = np.zeros((num_person, num_frame, num_keypoint),
                               dtype=np.float16)
    for i, poses in enumerate(pose_results):
        if len(poses) < 1:
            continue
        _keypoint = poses['keypoints']  # shape = (num_person, num_keypoint, 3)
        for j, pose in enumerate(_keypoint):
            keypoints[j, i] = pose[:, :2]
            keypoints_score[j, i] = pose[:, 2]

    fake_anno['keypoint'] = keypoints
    fake_anno['keypoint_score'] = keypoints_score

    results = video_cls_predictor([fake_anno])

    action_label = results[0]['class_name'][0]
    print(f'action label: {action_label}')

    vis_frames = [
        pose_predictor.show_result(original_frames[i], pose_results[i])
        if len(pose_results[i]) > 0 else original_frames[i]
        for i in range(num_frame)
    ]
    for frame in vis_frames:
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_file, remove_temp=True)
    print(f'Write video to {args.out_file} successfully!')

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
