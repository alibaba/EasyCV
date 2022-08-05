# Copyright (c) Alibaba, Inc. and its affiliates.
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from easycv.utils.metric_distance import (CosineSimilarity,
                                          DotproductSimilarity, LpDistance)
from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


@EVALUATORS.register_module
class RetrivalTopKEvaluator(Evaluator):
    """ RetrivalTopK evaluator,
      Retrival evaluate do the topK retrival, by measuring the distance of every 1 vs other.
      get the topK nearest, and count the match of ID. if Retrival = 1, Miss = 0. Finally average all
      RetrivalRate.
  """

    def __init__(self,
                 topk=(1, 2, 4, 8),
                 norm=0,
                 metric='cos',
                 pca=0,
                 dataset_name=None,
                 metric_names=['R@K=1'],
                 save_results=False,
                 save_results_dir='',
                 feature_keyword=['neck']):
        '''
        Args:
            top_k: tuple of int, evaluate top_k acc
        '''
        self._topk = topk
        self.norm = 0
        self.metric = metric
        self.dataset_name = dataset_name
        self.pca = pca

        self.save_results = save_results
        self.save_results_dir = Path(save_results_dir)
        if self.save_results:
            assert self.save_results_dir != '', 'when save retrival results formatted as .npy, save_results_dir should be set in config'

        self.feature_keyword = feature_keyword
        super(RetrivalTopKEvaluator, self).__init__(dataset_name, metric_names)

    def _evaluate_impl(self, results, gt_label, step=100):
        res = {}
        for key in self.feature_keyword:
            torch.cuda.empty_cache()
            res1 = self._evaluate(
                results, gt_label, step=100, feature_keyword=key)
            res.update(res1)

        return res

    def _evaluate(self, results, gt_label, step=100, feature_keyword='neck'):
        """Retrival evaluate do the topK retrival, by measuring the distance of every 1 vs other.
            get the topK nearest, and count the match of ID. if Retrival = 1, Miss = 0. Finally average all
            RetrivalRate.

        """
        # first print() is to show shape clearly in multi-process situation. don't comment it
        print()

        print('retrieval available keys : ', results.keys())
        results = results[feature_keyword]
        gt_label = gt_label

        if len(results.shape) > 2:
            results = results.view(results.shape[0], -1)

        print("retrieval results' shape:", results.shape)
        print('ground truth label shape:', gt_label.shape)

        if self.pca > 0:
            print(self.pca)
            pca_model = PCA(n_components=self.pca)
            c_results = results.cpu().detach().numpy()
            pca_model.fit(c_results)
            results_numpy = pca_model.transform(c_results)
            results = torch.from_numpy(results_numpy).to(gt_label.device)
            print('After pca')
            print(results.shape)

        assert results.size(0) == gt_label.size(0)

        # add GPU resouce to speed up
        try:
            results = results.cuda()
            gt_label = gt_label.cuda()
        except:
            pass

        if self.norm > 0:
            results = torch.nn.functional.normalize(
                results, p=self.norm, dim=1)
            gt_label = torch.nn.functional.normalize(
                gt_label, p=self.norm, dim=1)

        distance_dict = {
            'cos': CosineSimilarity,
            'dot': DotproductSimilarity,
            'lp': LpDistance,
        }

        distance_mult = {
            'cos': -1,
            'dot': -1,
            'lp': 1,
        }

        distance = distance_dict[self.metric]

        topk_list = self._topk
        topk_res = {}

        if type(topk_list) == int:
            topk_list = [topk_list]

        for k in topk_list:
            topk_res[k] = 0

        if self.save_results:
            retrival_index_results = torch.zeros([1, max(topk_list)],
                                                 dtype=torch.int64).to(
                                                     results.device)
            retrival_distance_results = torch.zeros([1, max(topk_list)],
                                                    dtype=torch.float32).to(
                                                        results.device)
            retrival_topk_results = torch.zeros(
                [results.size(0), len(topk_list)], dtype=torch.int8)

        for start_idx in tqdm(range(0, results.size(0), step)):
            dis_matrix = distance_mult[self.metric] * distance(
                results[start_idx:start_idx + step, ...], results)
            # this diag to eliminate distance with its self, which should be smallest, so we add max to kill it
            diag = torch.ones(dis_matrix.size(0)).to(
                dis_matrix.device) * torch.max(dis_matrix)
            diag = diag.to(dis_matrix.device)
            dis_matrix[range(dis_matrix.size(0)),
                       range(start_idx, start_idx + dis_matrix.size(0))] = diag

            query_distance, query_index = torch.topk(
                dis_matrix, k=max(topk_list), dim=1, largest=False)

            if self.save_results:
                retrival_index_results = torch.cat(
                    (retrival_index_results, query_index), 0)
                retrival_distance_results = torch.cat(
                    (retrival_distance_results, query_distance), 0)

            for idx in range(dis_matrix.size(0)):
                gt_query = torch.index_select(
                    gt_label, dim=0, index=query_index[idx]).cpu().numpy()
                for topk in topk_list:
                    gt_query_k = gt_query[:topk]
                    catch = False
                    for qi in range(topk):
                        if gt_query_k[qi] == gt_label[start_idx + idx]:
                            catch = True
                            if self.save_results:
                                retrival_topk_results[
                                    start_idx + idx,
                                    topk_list.index(topk)] = 1
                            break

                    if catch:
                        topk_res[topk] += 1

        if self.save_results:
            retrival_index_results = retrival_index_results[1:, ].cpu().numpy()
            retrival_distance_results = retrival_distance_results[
                1:, ].cpu().numpy()
            retrival_topk_results = retrival_topk_results.cpu().numpy()
            save_results_dict = {}
            save_results_dict[
                'retrival_index_results'] = retrival_index_results
            save_results_dict[
                'retrival_distance_results'] = retrival_distance_results
            save_results_dict['retrival_topk_results'] = retrival_topk_results
            save_results_dict['gt_label'] = gt_label.cpu().numpy()
            if self.dataset_name is not None:
                file_name = self.save_results_dir / '{}_retrival_results.npy'.format(
                    self.dataset_name)
            else:
                file_name = self.save_results_dir / 'retrival_results.npy'
            np.save(file_name, save_results_dict)

        eval_res = {}

        for k in topk_list:
            key_name = 'R@K={}'.format(k)
            eval_res[key_name] = (float(topk_res[k]) / results.size(0)) * 100

        print("Retrieval Eval of %s 's %s feature Result :" %
              (self.dataset_name, feature_keyword))
        print(eval_res)

        return eval_res


METRICS.register_default_best_metric(RetrivalTopKEvaluator, 'R@K=1', 'max')
