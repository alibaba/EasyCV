# Modified from https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/ppocr/metrics
import string
from collections import namedtuple

import numpy as np
from rapidfuzz.distance import Levenshtein
from shapely.geometry import Polygon

from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


@EVALUATORS.register_module()
class OCRDetEvaluator(Evaluator):

    def __init__(self, dataset_name=None, metric_names=['hmean']):
        self.iou_constraint = 0.5
        self.area_precision_constraint = 0.5
        super().__init__(dataset_name, metric_names)

    def _evaluate_impl(self, gt, pred):

        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        def compute_ap(confList, matchList, numGtCare):
            correct = 0
            AP = 0
            if len(confList) > 0:
                confList = np.array(confList)
                matchList = np.array(matchList)
                sorted_ind = np.argsort(-confList)
                confList = confList[sorted_ind]
                matchList = matchList[sorted_ind]
                for n in range(len(confList)):
                    match = matchList[n]
                    if match:
                        correct += 1
                        AP += float(correct) / (n + 1)

                if numGtCare > 0:
                    AP /= numGtCare

            return AP

        perSampleMetrics = {}

        matchedSum = 0

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        numGlobalCareGt = 0
        numGlobalCareDet = 0

        arrGlobalConfidences = []
        arrGlobalMatches = []

        recall = 0
        precision = 0
        hmean = 0

        detMatched = 0

        iouMat = np.empty([1, 1])

        gtPols = []
        detPols = []

        gtPolPoints = []
        detPolPoints = []

        # Array of Ground Truth Polygons' keys marked as don't Care
        gtDontCarePolsNum = []
        # Array of Detected Polygons' matched with a don't Care GT
        detDontCarePolsNum = []

        pairs = []
        detMatchedNums = []

        arrSampleConfidences = []
        arrSampleMatch = []

        evaluationLog = ''

        for n in range(len(gt)):
            points = gt[n]['points']
            # transcription = gt[n]['text']
            dontCare = gt[n]['ignore']
            #             points = Polygon(points)
            #             points = points.buffer(0)
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += 'GT polygons: ' + str(len(gtPols)) + (
            ' (' + str(len(gtDontCarePolsNum)) +
            " don't care)\n" if len(gtDontCarePolsNum) > 0 else '\n')

        for n in range(len(pred)):
            points = pred[n]['points']
            #             points = Polygon(points)
            #             points = points.buffer(0)
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        evaluationLog += 'DET polygons: ' + str(len(detPols)) + (
            ' (' + str(len(detDontCarePolsNum)) +
            " don't care)\n" if len(detDontCarePolsNum) > 0 else '\n')

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += 'Match GT #' + \
                                             str(gtNum) + ' with Det #' + str(detNum) + '\n'

        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(
                detMatched) / numDetCare

        hmean = 0 if (precision +
                      recall) == 0 else 2.0 * precision * recall / (
                          precision + recall)

        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'detMatched': detMatched,
        }
        return perSampleMetrics

    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
            methodRecall + methodPrecision)
        # print(methodRecall, methodPrecision, methodHmean)
        # sys.exit(-1)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        return methodMetrics

    def evaluate(self, preds, gt_polyons_batch, ignore_tags_batch, **kwargs):
        results = []
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred]
            result = self._evaluate_impl(gt_info_list, det_info_list)
            results.append(result)
        results = self.combine_results(results)
        return results


@EVALUATORS.register_module()
class OCRRecEvaluator(Evaluator):

    def __init__(self,
                 is_filter=False,
                 ignore_space=True,
                 dataset_name=None,
                 metric_names=['acc']):
        super().__init__(dataset_name, metric_names)
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters),
                   text))
        return text.lower()

    def _evaluate_impl(self, preds, labels, **kwargs):
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(' ', '')
                target = target.replace(' ', '')
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps)
        }


METRICS.register_default_best_metric(OCRDetEvaluator, 'hmean', 'max')
METRICS.register_default_best_metric(OCRRecEvaluator, 'acc', 'max')
