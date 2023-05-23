# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import json
import os
import unittest
import numpy as np
import time
import cv2
import torch
import scipy.io
from easycv.predictors.reid_predictor import ReIDPredictor
from tests.ut_config import SMALL_MARKET1501
from numpy.testing import assert_array_almost_equal


class ReIDPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def evaluate(self, qf, ql, qc, gf, gl, gc):
        query = qf.view(-1, 1)
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1]
        # good index
        query_index = np.argwhere(gl == ql)
        camera_index = np.argwhere(gc == qc)

        good_index = np.setdiff1d(
            query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl == -1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1)

        CMC_tmp = self.compute_mAP(index, good_index, junk_index)
        return CMC_tmp

    def compute_mAP(self, index, good_index, junk_index):
        ap = 0
        cmc = torch.IntTensor(len(index)).zero_()
        if good_index.size == 0:  # if empty
            cmc[0] = -1
            return ap, cmc

        # remove junk_index
        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]

        # find good_index index
        ngood = len(good_index)
        mask = np.in1d(index, good_index)
        rows_good = np.argwhere(mask == True)
        rows_good = rows_good.flatten()

        cmc[rows_good[0]:] = 1
        for i in range(ngood):
            d_recall = 1.0 / ngood
            precision = (i + 1) * 1.0 / (rows_good[i] + 1)
            if rows_good[i] != 0:
                old_precision = i * 1.0 / rows_good[i]
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2

        return ap, cmc

    def test(self):
        test_dir = os.path.join(SMALL_MARKET1501, 'pytorch')
        checkpoint = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/tracking/reid_r50_epoch_60_export.pt'
        gallery_dir = os.path.join(test_dir, 'gallery')
        query_dir = os.path.join(test_dir, 'query')

        # build model
        model = ReIDPredictor(
            model_path=checkpoint, config_file=None, batch_size=256)

        # extract features
        since = time.time()
        gallery_results = model(gallery_dir)
        query_results = model(query_dir)
        gallery_feature, gallery_cam, gallery_label = gallery_results[
            'img_feature'], gallery_results['img_cam'], gallery_results[
                'img_label']
        query_feature, query_cam, query_label = query_results[
            'img_feature'], query_results['img_cam'], query_results[
                'img_label']
        print(gallery_feature.size(), query_feature.size())
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        inference_result = './pytorch_result.mat'
        result = {
            'gallery_f': gallery_feature.numpy(),
            'gallery_label': gallery_label,
            'gallery_cam': gallery_cam,
            'query_f': query_feature.numpy(),
            'query_label': query_label,
            'query_cam': query_cam
        }
        scipy.io.savemat(inference_result, result)

        result = scipy.io.loadmat(inference_result)
        query_feature = torch.FloatTensor(result['query_f'])
        query_cam = result['query_cam'][0]
        query_label = result['query_label'][0]
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_cam = result['gallery_cam'][0]
        gallery_label = result['gallery_label'][0]

        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = self.evaluate(query_feature[i], query_label[i],
                                            query_cam[i], gallery_feature,
                                            gallery_label, gallery_cam)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp

        CMC = CMC.float()
        CMC = CMC / len(query_label)  # average CMC
        mAP = ap / len(query_label)
        assert_array_almost_equal(
            CMC[:10].tolist(),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            decimal=1)
        assert_array_almost_equal(mAP, 0.9925018971878582)
