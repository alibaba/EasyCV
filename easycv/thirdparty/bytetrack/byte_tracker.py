import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

try:
    from thirdparty.bytetrack.kalman_filter import KalmanFilter
    from thirdparty.bytetrack.basetrack import BaseTrack, TrackState
    from thirdparty.bytetrack import matching
except:
    from easycv.thirdparty.bytetrack.kalman_filter import KalmanFilter
    from easycv.thirdparty.bytetrack.basetrack import BaseTrack, TrackState
    from easycv.thirdparty.bytetrack import matching   

from easycv.predictors.builder import build_predictor, PREDICTORS


def post_process(bbox_xyxy, bbox_confidences, bbox_classes, target_label, threshold=None):
    # post process to filter result
    bbox_xyxy_tmp = []
    bbox_confidences_tmp = []
    bbox_classes_tmp = []
    assert len(target_label)==len(threshold), "detection post process, class filter need target_label and threshold both, and should be same length!"

    for bidx, bcls in enumerate(bbox_classes):
        if bcls in target_label and bbox_confidences[bidx] > threshold[target_label.index(bcls)]:
            bbox_xyxy_tmp.append(bbox_xyxy[bidx])
            bbox_confidences_tmp.append(bbox_confidences[bidx])
            bbox_classes_tmp.append(bbox_classes[bidx])
    bbox_xyxy = np.array(bbox_xyxy_tmp)
    bbox_confidences = np.array(bbox_confidences_tmp)
    bbox_classes = np.array(bbox_classes_tmp)
    return bbox_xyxy, bbox_confidences, bbox_classes



class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, 
        det_high_thresh=0.7,
        det_low_thresh=0.1, 
        match_thresh=0.8,
        match_thresh_second=1.0, 
        match_thresh_init=1.0, 
        track_buffer=5, 
        frame_rate=25):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_thresh = det_high_thresh
        self.match_thresh_second = match_thresh_second
        self.match_thresh_init = match_thresh_init
        self.det_thresh = det_high_thresh
        self.match_thresh = match_thresh
        self.low_thresh = det_low_thresh

        self.buffer_size = int(frame_rate / 30 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    
    def update(self, bbox_xyxy, confidences, classes, target_label=None, target_threshold=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        bboxes = bbox_xyxy
        scores = confidences
        classes = classes

        if target_label is not None:
            boxes, scores, classes = post_process(bboxes, confidences, classes, target_label=target_label, threshold=target_threshold)

        remain_inds = scores > self.track_thresh
        inds_low = scores > self.low_thresh
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.match_thresh_second)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh_init)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        fres = []
        for t in output_stracks:
            tid = t.track_id
            tlbr = t.tlwh_to_tlbr(t.tlwh)
            tlbr = [int(i) for i in tlbr]
            fres.append(tlbr + [tid])

        return np.array(fres)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


if __name__ == "__main__":
    import cv2
    import random
    from PIL import Image
    import  argparse

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(2000)]
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        '''
        yolo used plot 
        :x : bboxes
        :img: ploted image
        :color: color 3
        :label: label text
        :return: None
        '''
        # Plots one bounding box on image img
        tl = int(line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2)) + 1  # line/font thickness
        # tl = int(line_thickness)
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 10, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, 0.5, [225, 255, 255], thickness=max(tf-10,2), lineType=cv2.LINE_AA)
        return



    parser = argparse.ArgumentParser('ev eas processor local runner')
    parser.add_argument('--test_video', type=str, help='local model dir')
    parser.add_argument('--det_model', type=str,  help='local model dir')


    args = parser.parse_args()

    # video_list : oss://pai-vision-data-inner/data/yuanqisenlin/poc_0917/video_data/正常拿取商品/
    test_video = args.test_video
    print("test video         : ", test_video)

    #build tracker
    
    # custom application
    # YoloDetector + FeatureExtractor
    tracker = BYTETracker(
        detection_model_path=args.det_model,
        detection_model_type='TorchYolo5Predictor',
        det_high_thresh=0.2,
        det_low_thresh=0.05, 
        match_thresh=1.0,
        match_thresh_second=1.0, 
        match_thresh_init=1.0,  
        track_buffer=2, 
        frame_rate=25)


    # read input video
    cap = cv2.VideoCapture(test_video)
    img_list = []
    ret = True
    while(cap.isOpened() and ret):
        ret, frame = cap.read()
        if ret:
            img_list.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    res = tracker.predict(img_list, target_label=[2], target_threshold=[0.1])
    res_keys = list(res.keys())
    res_keys = sorted(res_keys)
    for idx in res_keys:
        tracks = res[idx]
        img = np.array(img_list[idx])
        if len(tracks) > 0:
            for t in tracks:
                tid = t[-1]
                box = t[:4]
                box = [int(i) for i in box]
                plot_one_box(box, img, color=colors[int(tid)], label=str(tid))
        img_list[idx] = img

    #write out video
    if test_video[-4:] == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if test_video[-4:] == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    w, h = img_list[0].shape[1], img_list[0].shape[0]
    print("video w, h         : ", w,h)
    # output_video = test_video.replace(test_video[-4:], '_output'+test_video[-4:])
    output_video = "test.mp4"
    print("output video       : ", output_video)
    videowriter = cv2.VideoWriter(output_video, fourcc, 15, (w, h))
    for idx, img in enumerate(img_list):
        img = np.asarray(img)
        img = img[:,:,[2,1,0]]
        videowriter.write(img)
