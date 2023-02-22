# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
import os
import random
import re
import sys
from functools import reduce

import cv2
import mmcv
import numpy as np
import torch
from torchvision.transforms import functional as F


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_tracking(image,
                  tlwhs,
                  obj_ids,
                  scores=None,
                  frame_id=0,
                  fps=0.,
                  ids2names=[],
                  do_entrance_counting=False,
                  entrance=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(0.5, image.shape[1] / 3000.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    cv2.putText(
        im,
        'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
        (0, int(15 * text_scale) + 5),
        cv2.FONT_ITALIC,
        text_scale, (0, 0, 255),
        thickness=text_thickness)
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = 'ID: {}'.format(int(obj_id))
        if ids2names != []:
            assert len(
                ids2names) == 1, 'plot_tracking only supports single classes.'
            id_text = 'ID: {}_'.format(ids2names[0]) + id_text
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(
            im,
            intbox[0:2],
            intbox[2:4],
            color=color,
            thickness=line_thickness)
        cv2.putText(
            im,
            id_text, (intbox[0], intbox[1] - 25),
            cv2.FONT_ITALIC,
            text_scale, (0, 255, 255),
            thickness=text_thickness)

        if scores is not None:
            text = 'score: {:.2f}'.format(float(scores[i]))
            cv2.putText(
                im,
                text, (intbox[0], intbox[1] - 6),
                cv2.FONT_ITALIC,
                text_scale, (0, 255, 0),
                thickness=text_thickness)
    if do_entrance_counting:
        entrance_line = tuple(map(int, entrance))
        cv2.rectangle(
            im,
            entrance_line[0:2],
            entrance_line[2:4],
            color=(0, 255, 255),
            thickness=line_thickness)
    return im


def get_mtmct_matching_results(pred_mtmct_file,
                               secs_interval=0.5,
                               video_fps=20):
    res = np.loadtxt(pred_mtmct_file)  # 'cid, tid, fid, x1, y1, w, h, -1, -1'
    camera_ids = list(map(int, np.unique(res[:, 0])))

    res = res[:, :7]
    # each line in res: 'cid, tid, fid, x1, y1, w, h'

    camera_tids = []
    camera_results = dict()
    for c_id in camera_ids:
        camera_results[c_id] = res[res[:, 0] == c_id]
        tids = np.unique(camera_results[c_id][:, 1])
        tids = list(map(int, tids))
        camera_tids.append(tids)

    # select common tids throughout each video
    common_tids = reduce(np.intersect1d, camera_tids)
    if len(common_tids) == 0:
        print(
            'No common tracked ids in these videos, please check your MOT result or select new videos.'
        )
        return None, None

    # get mtmct matching results by cid_tid_fid_results[c_id][t_id][f_id]
    cid_tid_fid_results = dict()
    cid_tid_to_fids = dict()
    interval = int(secs_interval * video_fps)  # preferably less than 10
    for c_id in camera_ids:
        cid_tid_fid_results[c_id] = dict()
        cid_tid_to_fids[c_id] = dict()
        for t_id in common_tids:
            tid_mask = camera_results[c_id][:, 1] == t_id
            cid_tid_fid_results[c_id][t_id] = dict()

            camera_trackid_results = camera_results[c_id][tid_mask]
            fids = np.unique(camera_trackid_results[:, 2])
            fids = fids[fids % interval == 0]
            fids = list(map(int, fids))
            cid_tid_to_fids[c_id][t_id] = fids

            for f_id in fids:
                st_frame = f_id
                ed_frame = f_id + interval

                st_mask = camera_trackid_results[:, 2] >= st_frame
                ed_mask = camera_trackid_results[:, 2] < ed_frame
                frame_mask = np.logical_and(st_mask, ed_mask)
                cid_tid_fid_results[c_id][t_id][f_id] = camera_trackid_results[
                    frame_mask]

    return camera_results, cid_tid_fid_results


def save_mtmct_crops(cid_tid_fid_res,
                     images_dir,
                     crops_dir,
                     width=300,
                     height=200):
    camera_ids = cid_tid_fid_res.keys()
    seqs_folder = os.listdir(images_dir)
    seqs = []
    for x in seqs_folder:
        if os.path.isdir(os.path.join(images_dir, x)):
            seqs.append(x)
    assert len(seqs) == len(camera_ids)
    seqs.sort()

    if not os.path.exists(crops_dir):
        os.makedirs(crops_dir)

    common_tids = list(cid_tid_fid_res[list(camera_ids)[0]].keys())

    # get crops by name 'tid_cid_fid.jpg
    for t_id in common_tids:
        for i, c_id in enumerate(camera_ids):
            infer_dir = os.path.join(images_dir, seqs[i])
            if os.path.exists(os.path.join(infer_dir, 'img1')):
                infer_dir = os.path.join(infer_dir, 'img1')
            all_images = os.listdir(infer_dir)
            all_images.sort()

            for f_id in cid_tid_fid_res[c_id][t_id].keys():
                frame_idx = f_id - 1 if f_id > 0 else 0
                im_path = os.path.join(infer_dir, all_images[frame_idx])

                im = cv2.imread(im_path)  # (H, W, 3)

                # only select one track
                track = cid_tid_fid_res[c_id][t_id][f_id][0]

                cid, tid, fid, x1, y1, w, h = [int(v) for v in track]
                clip = im[y1:(y1 + h), x1:(x1 + w)]
                clip = cv2.resize(clip, (width, height))

                cv2.imwrite(
                    os.path.join(
                        crops_dir, 'tid{:06d}_cid{:06d}_fid{:06d}.jpg'.format(
                            tid, cid, fid)), clip)

            print(
                'Finish cropping image of tracked_id {} in camera: {}'.format(
                    t_id, c_id))


def save_mtmct_vis_results(camera_results,
                           images_dir,
                           save_dir,
                           save_videos=False):
    # camera_results: 'cid, tid, fid, x1, y1, w, h'
    camera_ids = camera_results.keys()
    seqs_folder = os.listdir(images_dir)
    seqs = []
    for x in seqs_folder:
        if os.path.isdir(os.path.join(images_dir, x)):
            seqs.append(x)
    assert len(seqs) == len(camera_ids)
    seqs.sort()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, c_id in enumerate(camera_ids):
        print('Start visualization for camera {} of sequence {}.'.format(
            c_id, seqs[i]))
        cid_save_dir = os.path.join(save_dir, '{}'.format(seqs[i]))
        if not os.path.exists(cid_save_dir):
            os.makedirs(cid_save_dir)

        infer_dir = os.path.join(images_dir, seqs[i])
        if os.path.exists(os.path.join(infer_dir, 'img1')):
            infer_dir = os.path.join(infer_dir, 'img1')
        all_images = os.listdir(infer_dir)
        all_images.sort()

        for f_id, im_path in enumerate(all_images):
            img = cv2.imread(os.path.join(infer_dir, im_path))
            tracks = camera_results[c_id][camera_results[c_id][:, 2] == f_id]
            if tracks.shape[0] > 0:
                tracked_ids = tracks[:, 1]
                xywhs = tracks[:, 3:]
                online_im = plot_tracking(
                    img, xywhs, tracked_ids, scores=None, frame_id=f_id)
            else:
                online_im = img
                print('Frame {} of seq {} has no tracking results'.format(
                    f_id, seqs[i]))

            cv2.imwrite(
                os.path.join(cid_save_dir, '{:05d}.jpg'.format(f_id)),
                online_im)
            if f_id % 40 == 0:
                print('Processing frame {}'.format(f_id))

        if save_videos:
            output_video_path = os.path.join(
                cid_save_dir, '..', '{}_mtmct_vis.mp4'.format(seqs[i]))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(
                cid_save_dir, output_video_path)
            os.system(cmd_str)
            print('Save camera {} video in {}.'.format(seqs[i],
                                                       output_video_path))


def parse_pt_gt(mot_feature):
    img_rects = dict()
    for line in mot_feature:
        fid = int(re.sub('[a-z,A-Z]', '', mot_feature[line]['frame']))
        tid = mot_feature[line]['id']
        rect = list(map(lambda x: int(float(x)), mot_feature[line]['bbox']))
        if fid not in img_rects:
            img_rects[fid] = list()
        rect.insert(0, tid)
        img_rects[fid].append(rect)
    return img_rects


def gen_res(output_dir_filename, scene_cluster, map_tid, mot_list_breaks):
    f_w = open(output_dir_filename, 'w')
    for idx, mot_feature in enumerate(mot_list_breaks):
        cid = scene_cluster[idx]
        img_rects = parse_pt_gt(mot_feature)

        for fid in img_rects:
            tid_rects = img_rects[fid]
            fid = int(fid) + 1
            for tid_rect in tid_rects:
                tid = tid_rect[0]
                rect = tid_rect[1:]
                cx = 0.5 * rect[0] + 0.5 * rect[2]
                cy = 0.5 * rect[1] + 0.5 * rect[3]
                w = rect[2] - rect[0]
                w = min(w * 1.2, w + 40)
                h = rect[3] - rect[1]
                h = min(h * 1.2, h + 40)
                rect[2] -= rect[0]
                rect[3] -= rect[1]
                rect[0] = max(0, rect[0])
                rect[1] = max(0, rect[1])
                x1, y1 = max(0, cx - 0.5 * w), max(0, cy - 0.5 * h)
                x2, y2 = cx + 0.5 * w, cy + 0.5 * h
                w, h = x2 - x1, y2 - y1
                new_rect = list(map(int, [x1, y1, w, h]))
                rect = list(map(int, rect))
                if (cid, tid) in map_tid:
                    new_tid = map_tid[(cid, tid)]
                    f_w.write(
                        str(cid) + ' ' + str(new_tid) + ' ' + str(fid) + ' ' +
                        ' '.join(map(str, new_rect)) + ' -1 -1'
                        '\n')
    print('gen_res: write file in {}'.format(output_dir_filename))
    f_w.close()


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


def normalize(nparray, axis=0):
    try:
        from sklearn import preprocessing
    except Exception as e:
        raise RuntimeError(
            'Unable to use sklearn in MTMCT in PP-Tracking, please install sklearn, for example: `pip install sklearn`'
        )
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray


def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][0] == cid_tids[j][0]:
                st_mask[i, j] = 0.
    return st_mask


def visual_rerank(prb_feats,
                  gal_feats,
                  cid_tids,
                  use_ff=False,
                  use_rerank=False):
    """Rerank by visual cures."""
    gal_labels = np.array([[0, item[0]] for item in cid_tids])
    prb_labels = gal_labels.copy()
    sims = 1.0 - np.dot(prb_feats, gal_feats.T)

    # NOTE: sims here is actually dist, the smaller the more similar
    return 1.0 - sims


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
    assert (len(scene_cluster) != 0), 'Error: scene_cluster length equals 0'
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
        ' -i ', video_path, ' -r ',
        str(frame_rate), ' -f image2 ', outformat
    ]
    cmd = ''.join(cmd) + _dict2str(kargs)

    if os.system(cmd) != 0:
        raise RuntimeError('ffmpeg process video: {} error'.format(video_path))
        sys.exit(-1)

    sys.stdout.flush()
    return out_full_path


def parse_pt(mot_feature):
    mot_list = dict()
    for line in mot_feature:
        fid = int(re.sub('[a-z,A-Z]', '', mot_feature[line]['frame']))
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


def prepare_crop_imgs(detection_results):
    img_metas = detection_results['img_metas']
    detection_boxes = detection_results['boxes']  # id, x0, y0, x1, y1, score
    pred_xyxys = detection_boxes[:, 1:5]

    ori_img_shape = img_metas['ori_img_shape'][:2]
    pred_xyxys, keep_idx = clip_box(pred_xyxys, ori_img_shape)

    if len(keep_idx[0]) == 0:
        return None

    detection_boxes = detection_boxes[keep_idx[0]]
    pred_xyxys = detection_boxes[:, 1:5]

    filename = img_metas['filename']
    w, h = img_metas['batch_input_shape']
    ori_image = decode_image(filename)
    batch_crop_imgs = get_crops(pred_xyxys, ori_image, w, h)

    return batch_crop_imgs, detection_boxes


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


def detection_result_filter(bboxes,
                            scores,
                            classes,
                            target_classes,
                            target_thresholds=None):
    # post process to filter result
    bboxes_tmp = []
    scores_tmp = []
    classes_tmp = []
    assert len(target_classes) == len(
        target_thresholds
    ), 'detection post process, class filter need target_classes and target_thresholds both, and should be same length!'

    for bidx, bcls in enumerate(classes):
        if bcls in target_classes and scores[bidx] > target_thresholds[
                target_classes.index(bcls)]:
            bboxes_tmp.append(bboxes[bidx])
            scores_tmp.append(scores[bidx])
            classes_tmp.append(classes[bidx])
    bboxes = np.array(bboxes_tmp)
    scores = np.array(scores_tmp)
    classes = np.array(classes_tmp)
    return bboxes, scores, classes


def results2outs(bbox_results=None, **kwargs):
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
    import seaborn as sns
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
            cv2.rectangle(
                img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

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


def show_result(img,
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