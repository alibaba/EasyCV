# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from easycv.utils.logger import print_log
from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


@EVALUATORS.register_module
class ClsEvaluator(Evaluator):
    """ Classification evaluator.
  """

    def __init__(self,
                 topk=(1, 5),
                 dataset_name=None,
                 metric_names=['neck_top1'],
                 neck_num=None,
                 class_list=None):
        '''

        Args:
            top_k (int, tuple): int or tuple of int, evaluate top_k acc
            dataset_name: eval dataset name
            metric_names: eval metrics name
            neck_num: some model contains multi-neck to support multitask, neck_num means use the no.neck_num  neck output of model to eval
        '''
        if isinstance(topk, int):
            topk = (topk, )
        self._topk = topk
        self.dataset_name = dataset_name
        self.neck_num = neck_num
        self.class_list = class_list

        super(ClsEvaluator, self).__init__(dataset_name, metric_names)

    def _evaluate_impl(self, predictions, gt_labels):
        ''' python evaluation code which will be run after all test batched data are predicted

        Args:
            predictions: dict of tensor with shape NxC, from each cls heads
            gt_labels: int32 tensor with shape N

        Return:
            a dict,  each key is metric_name, value is metric value
        '''
        eval_res = OrderedDict()

        if isinstance(gt_labels, dict):
            assert len(gt_labels) == 1
            gt_labels = list(gt_labels.values())[0]

        target = gt_labels.long()

        # if self.neck_num is not None:
        if self.neck_num is None:
            if len(predictions) > 1:
                predictions = {'neck': predictions['neck']}
        else:
            predictions = {
                'neck_%d_0' % self.neck_num:
                predictions['neck_%d_0' % self.neck_num]
            }

        for key, scores in predictions.items():
            assert scores.size(0) == target.size(0), \
                'Inconsistent length for results and labels, {} vs {}'.format(
                scores.size(0), target.size(0))
            num = scores.size(0)

            # Avoid topk values greater than the number of categories
            self._topk = np.array(list(self._topk))
            self._topk = np.clip(self._topk, 1, scores.shape[-1])

            _, pred = scores.topk(
                max(self._topk), dim=1, largest=True, sorted=True)

            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN
            for k in self._topk:
                # use contiguous() to avoid eval view failed
                correct_k = correct[:k].contiguous().view(-1).float().sum(
                    0).item()
                acc = correct_k * 100.0 / num
                eval_res['{}_top{}'.format(key, k)] = acc

            if self.class_list is not None:
                # confusion_matrix
                class_num = scores.shape[1]
                tp = np.zeros(class_num)  # predict: 1, target: 1
                fn = np.zeros(class_num)  # predict: 0, target: 1
                fp = np.zeros(class_num)  # predict: 1, target: 0
                tn = np.zeros(class_num)  # predict: 0, target: 0
                attend = np.zeros(class_num)  # target num
                valid_true = []
                valid_pred = []

                target_onehot = torch.zeros([scores.shape[0], scores.shape[1]],
                                            dtype=scores.dtype,
                                            layout=scores.layout,
                                            device=scores.device)
                target_onehot.scatter_(1, target.unsqueeze(-1), 1)
                predict_onehot = torch.zeros(
                    [scores.shape[0], scores.shape[1]],
                    dtype=scores.dtype,
                    layout=scores.layout,
                    device=scores.device)
                predict_onehot.scatter_(
                    1,
                    torch.argmax(scores, dim=1).unsqueeze(-1), 1)

                target_onehot = target_onehot.numpy()
                predict_onehot = predict_onehot.numpy()

                tp += np.sum((predict_onehot == target_onehot), axis=0)
                fn += np.sum((target_onehot - predict_onehot) > 0, axis=0)
                fp += np.sum((predict_onehot - target_onehot) > 0, axis=0)
                tn += np.sum(((predict_onehot == 0) & (target_onehot == 0)),
                             axis=0)
                tp -= np.sum(((predict_onehot == 0) & (target_onehot == 0)),
                             axis=0)
                attend += np.sum(target_onehot, axis=0)

                recall = tp / (tp + fn + 0.00001)
                precision = tp / (tp + fp + 0.00001)
                f1 = 2 * recall * precision / (recall + precision + 0.00001)

                recall_mean = np.mean(recall, axis=0)
                precision_mean = np.mean(precision)
                f1_mean = np.mean(f1, axis=0)

                valid_target = target_onehot[
                    np.sum(target_onehot, axis=1) <= 1]
                valid_predict = predict_onehot[
                    np.sum(target_onehot, axis=1) <= 1]
                for sub_predict, sub_target in zip(valid_target,
                                                   valid_predict):
                    valid_true.append(self.class_list[sub_target.argmax()])
                    valid_pred.append(self.class_list[sub_predict.argmax()])

                matrix = confusion_matrix(
                    valid_true, valid_pred, labels=self.class_list)

                # print_log(
                #     'recall:{}\nprecision:{}\nattend:{}\nTP:{}\nFN:{}\nFP:{}\nTN:{}\nrecall/mean:{}\nprecision/mean:{}\nF1/mean:{}\nconfusion_matrix:{}\n'
                #     .format(recall, precision, attend, tp, fn, fp, tn,
                #             recall_mean, precision_mean, f1_mean, matrix))

                eval_res[key] = \
                    'recall:{}\nprecision:{}\nattend:{}\nTP:{}\nFN:{}\nFP:{}\nTN:{}\nrecall/mean:{}\nprecision/mean:{}\nF1/mean:{}\nconfusion_matrix:{}\n'\
                    .format(recall, precision, attend, tp, fn, fp, tn, recall_mean, precision_mean, f1_mean, matrix.tolist())

        return eval_res


@EVALUATORS.register_module
class MultiLabelEvaluator(Evaluator):
    """ Multilabel Classification evaluator.
  """

    def __init__(self, dataset_name=None, metric_names=['mAP']):
        '''

        Args:
            dataset_name: eval dataset name
            metric_names: eval metrics name
        '''
        self.dataset_name = dataset_name

        super(MultiLabelEvaluator, self).__init__(dataset_name, metric_names)

    def _evaluate_impl(self, predictions, gt_labels):
        preds = torch.sigmoid(predictions['neck'])

        map_out = self.mAP(preds, gt_labels)
        eval_res = {
            'mAP': map_out,
        }
        return eval_res

    def mAP(self, pred, target):
        """Calculate the mean average precision with respect of classes.
        Args:
            pred (torch.Tensor | np.ndarray): The model prediction with shape
                (N, C), where C is the number of classes.
            target (torch.Tensor | np.ndarray): The target of each prediction with
                shape (N, C), where C is the number of classes. 1 stands for
                positive examples, 0 stands for negative examples and -1 stands for
                difficult examples.
        Returns:
            float: A single float as mAP value.
        """
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            pred = pred.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        elif not (isinstance(pred, np.ndarray)
                  and isinstance(target, np.ndarray)):
            raise TypeError('pred and target should both be torch.Tensor or'
                            'np.ndarray')

        assert pred.shape == \
            target.shape, 'pred and target should be in the same shape.'
        num_classes = pred.shape[1]
        ap = np.zeros(num_classes)
        mean_ap = ap.mean() * 100.0
        return mean_ap

    def average_precision(self, pred, target):
        r"""Calculate the average precision for a single class.
        AP summarizes a precision-recall curve as the weighted mean of maximum
        precisions obtained for any r'>r, where r is the recall:
        .. math::
            \text{AP} = \sum_n (R_n - R_{n-1}) P_n
        Note that no approximation is involved since the curve is piecewise
        constant.
        Args:
            pred (np.ndarray): The model prediction with shape (N, ).
            target (np.ndarray): The target of each prediction with shape (N, ).
        Returns:
            float: a single float as average precision value.
        """
        eps = np.finfo(np.float32).eps

        # sort examples
        sort_inds = np.argsort(-pred)
        sort_target = target[sort_inds]

        # count true positive examples
        pos_inds = sort_target == 1
        tp = np.cumsum(pos_inds)
        total_pos = tp[-1]

        # count not difficult examples
        pn_inds = sort_target != -1
        pn = np.cumsum(pn_inds)

        tp[np.logical_not(pos_inds)] = 0
        precision = tp / np.maximum(pn, eps)
        ap = np.sum(precision) / np.maximum(total_pos, eps)
        return ap


METRICS.register_default_best_metric(ClsEvaluator, 'neck_top1', 'max')
METRICS.register_default_best_metric(MultiLabelEvaluator, 'mAP', 'max')
