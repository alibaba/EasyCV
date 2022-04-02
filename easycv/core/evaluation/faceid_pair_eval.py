# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import sklearn
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,
                 threshold_idx], fprs[fold_idx,
                                      threshold_idx], _ = calculate_accuracy(
                                          threshold, dist[test_set],
                                          actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(
            np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def faceid_evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    '''
    Do Kfold=nrof_folds faceid pair-match test for embeddings
    Args:
      embeddings: [N x C] inputs embedding of all dataset
      actual_issame: [N/2, 1] label of is match
      nrof_folds: KFold number
      pca : > 0 means, do pca and trans embedding to [N, pca] feature

    Return:
      KFold average best accuracy and best threshold
    '''

    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    # thresholds = np.arange(0, 0.1, 0.001)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds,
        embeddings1,
        embeddings2,
        np.asarray(actual_issame),
        nrof_folds=nrof_folds,
        pca=pca)
    # return tpr, fpr, accuracy, best_thresholds
    return accuracy.mean(), best_thresholds.mean()


@EVALUATORS.register_module
class FaceIDPairEvaluator(Evaluator):
    """ FaceIDPairEvaluator evaluator.
    Input nx2 pairs and label, kfold thresholds search and return average best accuracy
  """

    def __init__(self,
                 dataset_name=None,
                 metric_names=['acc'],
                 kfold=10,
                 pca=0):
        '''
        Faceid small dataset evaluator, do pair match validation
        Args:
            dataset_name : faceid small validate set name, include [lfw, agedb_30, cfp_ff, cfp_fw, calfw]
            kfold : Kfold for train/val split
            pca : pca dimensions, if > 0, do PCA for input feature, transfer to [n, pca]
        Return:
            None
    '''
        self.kfold = kfold
        self.pca = pca
        self.dataset_name = dataset_name
        super(FaceIDPairEvaluator, self).__init__(dataset_name, metric_names)

    def _evaluate_impl(self, predictions, gt_labels):
        ''' python evaluation code which will be run after all test batched data are predicted

        Args:
            predictions: dict of tensor with shape NxC, from each faceid pair feature (N/2) pairs
                gt_labels: tensor with shape Nx1

        Return:
            a dict,  each key is metric_name, value is metric value
        '''

        if type(predictions) == dict:
            predictions = predictions['neck']

        val = sklearn.preprocessing.normalize(predictions)
        accuracy, threshold = faceid_evaluate(val, gt_labels[0::2], self.kfold,
                                              self.pca)
        eval_res = {'acc': accuracy}
        return eval_res


METRICS.register_default_best_metric(FaceIDPairEvaluator, 'acc', 'max')
