# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
import cv2
import numpy as np
import random
import re
import seaborn as sns

import mmcv

from torchvision.transforms import functional as F

def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster

def combin_feature(cid_tid_dict, sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct) < 2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict

def get_cid_tid(cluster_labels, cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster

def get_sim_matrix(cid_tid_dict,
                   cid_tids,
                   use_ff=True,
                   use_rerank=True,
                   use_st_filter=False):
    # Note: camera independent get_sim_matrix function,
    # which is different from the one in camera_utils.py.
    count = len(cid_tids)

    q_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array(
        [cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)

    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)

    visual_sim_matrix = visual_rerank(
        q_arr, g_arr, cid_tids, use_ff=use_ff, use_rerank=use_rerank)
    visual_sim_matrix = visual_sim_matrix.astype('float32')

    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

def get_labels(cid_tid_dict,
               cid_tids,
               use_ff=True,
               use_rerank=True,
               use_st_filter=False):
    try:
        from sklearn.cluster import AgglomerativeClustering
    except Exception as e:
        raise RuntimeError(
            'Unable to use sklearn in MTMCT in PP-Tracking, please install sklearn, for example: `pip install sklearn`'
        )
    # 1st cluster
    sim_matrix = get_sim_matrix(
        cid_tid_dict,
        cid_tids,
        use_ff=use_ff,
        use_rerank=use_rerank,
        use_st_filter=use_st_filter)
    cluster_labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        affinity='precomputed',
        linkage='complete').fit_predict(1 - sim_matrix)
    labels = get_match(cluster_labels)
    sub_cluster = get_cid_tid(labels, cid_tids)

    # 2nd cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sim_matrix = get_sim_matrix(
        cid_tid_dict_new,
        cid_tids,
        use_ff=use_ff,
        use_rerank=use_rerank,
        use_st_filter=use_st_filter)
    cluster_labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.9,
        affinity='precomputed',
        linkage='complete').fit_predict(1 - sim_matrix)
    labels = get_match(cluster_labels)
    sub_cluster = get_cid_tid(labels, cid_tids)

    return labels

def sub_cluster(cid_tid_dict,
                scene_cluster,
                use_ff=False,
                use_rerank=False,
                use_st_filter=False):
    '''
    cid_tid_dict: all camera_id and track_id
    scene_cluster: like [41, 42, 43, 44, 45, 46] in AIC21 MTMCT S06 test videos
    '''
    assert (len(scene_cluster) != 0), "Error: scene_cluster length equals 0"
    cid_tids = sorted(
        [key for key in cid_tid_dict.keys() if key[0] in scene_cluster])
    clu = get_labels(
        cid_tid_dict,
        cid_tids,
        use_ff=use_ff,
        use_rerank=use_rerank,
        use_st_filter=use_st_filter)
    new_clu = list()
    for c_list in clu:
        if len(c_list) <= 1: continue
        cam_list = [cid_tids[c][0] for c in c_list]
        if len(cam_list) != len(set(cam_list)): continue
        new_clu.append([cid_tids[c] for c in c_list])
    all_clu = new_clu
    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    return cid_tid_label

def _is_valid_video(f, extensions=('.mp4', '.avi', '.mov', '.rmvb', '.flv')):
    return f.lower().endswith(extensions)

def video2frames(video_path, outpath, frame_rate=25, **kargs):
    def _dict2str(kargs):
        cmd_str = ''
        for k, v in kargs.items():
            cmd_str += (' ' + str(k) + ' ' + str(v))
        return cmd_str

    ffmpeg = ['ffmpeg ', ' -y -loglevel ', ' error ']
    vid_name = os.path.basename(video_path).split('.')[0]
    out_full_path = os.path.join(outpath, vid_name)

    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)

    # video file name
    outformat = os.path.join(out_full_path, '%05d.jpg')

    cmd = ffmpeg
    cmd = ffmpeg + [
        ' -i ', video_path, ' -r ', str(frame_rate), ' -f image2 ', outformat
    ]
    cmd = ''.join(cmd) + _dict2str(kargs)

    if os.system(cmd) != 0:
        raise RuntimeError('ffmpeg process video: {} error'.format(video_path))
        sys.exit(-1)

    sys.stdout.flush()
    return out_full_path

def parse_bias(cameras_bias):
    cid_bias = dict()
    for cameras in cameras_bias.keys():
        cameras_id = re.sub('[a-z,A-Z]', "", cameras)
        cameras_id = int(cameras_id)
        bias = cameras_bias[cameras]
        cid_bias[cameras_id] = float(bias)
    return cid_bias

def parse_pt(mot_feature):
    mot_list = dict()
    for line in mot_feature:
        fid = int(re.sub('[a-z,A-Z]', "", mot_feature[line]['frame']))
        tid = mot_feature[line]['id']
        bbox = list(map(lambda x: int(float(x)), mot_feature[line]['bbox']))
        if tid not in mot_list:
            mot_list[tid] = dict()
        out_dict = mot_feature[line]
        mot_list[tid][fid] = out_dict
    return mot_list

def gen_new_mot(mot_list):
    out_dict = dict()
    for tracklet in mot_list:
        tracklet = mot_list[tracklet]
        for f in tracklet:
            out_dict[tracklet[f]['imgname']] = tracklet[f]
    return out_dict

def trajectory_fusion(mot_feature, cid, cid_bias):
    cur_bias = cid_bias[cid]
    mot_list_break = {}

    mot_list = parse_pt(mot_feature)

    mot_list_break = gen_new_mot(mot_list)  # save break feature for gen result

    tid_data = dict()
    for tid in mot_list:
        tracklet = mot_list[tid]
        if len(tracklet) <= 1:
            continue
        frame_list = list(tracklet.keys())
        frame_list.sort()
        feature_list = [
            tracklet[f]['feat'] for f in frame_list
            if (tracklet[f]['bbox'][3] - tracklet[f]['bbox'][1]) *
            (tracklet[f]['bbox'][2] - tracklet[f]['bbox'][0]) > 2000
        ]
        if len(feature_list) < 2:
            feature_list = [tracklet[f]['feat'] for f in frame_list]
        io_time = [
            cur_bias + frame_list[0] / 10., cur_bias + frame_list[-1] / 10.
        ]
        all_feat = np.array([feat for feat in feature_list])
        mean_feat = np.mean(all_feat, axis=0)
        tid_data[tid] = {
            'cam': cid,
            'tid': tid,
            'mean_feat': mean_feat,
            'frame_list': frame_list,
            'tracklet': tracklet,
            'io_time': io_time
        }
    return tid_data, mot_list_break

def reid_predictor(detection_results, reid_model):
    img_metas = detection_results['img_metas']
    pred_dets = detection_results['boxes']  # id, x0, y0, x1, y1, score
    pred_xyxys = pred_dets[:, 1:5]

    ori_img_shape = img_metas['ori_img_shape'][:2]
    pred_xyxys, keep_idx = clip_box(pred_xyxys, ori_img_shape)

    if len(keep_idx[0]) == 0:
        return None

    pred_dets = pred_dets[keep_idx[0]]
    pred_xyxys = pred_dets[:, 1:5]

    filename = img_metas['filename']
    w, h = img_metas['batch_input_shape']
    ori_image = decode_image(filename)
    batch_crop_imgs = get_crops(pred_xyxys, ori_image, w, h)

    pred_embeddings = reid_model(batch_crop_imgs)

    return pred_embeddings

def decode_image(im_file):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    return im

def clip_box(xyxy, ori_img_shape):
    H, W = ori_img_shape
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], a_min=0, a_max=W)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], a_min=0, a_max=H)
    w = xyxy[:, 2:3] - xyxy[:, 0:1]
    h = xyxy[:, 3:4] - xyxy[:, 1:2]
    mask = np.logical_and(h > 0, w > 0)
    keep_idx = np.nonzero(mask)
    return xyxy[keep_idx[0]], keep_idx

def get_crops(xyxy, ori_img, w, h):
    crop_imgs = []
    xyxy = xyxy.astype(np.int64)
    for i, bbox in enumerate(xyxy):
        crop_img = ori_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        crop_img = cv2.resize(crop_img, (w, h))
        crop_imgs.append(np.array(crop_img))
    return crop_imgs

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