# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from nni.algorithms.hpo.medianstop_assessor import MedianstopAssessor
from nni.assessor import AssessResult
from nni.utils import extract_scalar_history
from typing_extensions import Literal


class PAIAssessor(MedianstopAssessor):

    def __init__(self,
                 optimize_mode: Literal['minimize', 'maximize'] = 'maximize',
                 start_step: int = 0,
                 moving_avg=False,
                 proportion=0.5,
                 patience=None):
        self._start_step = start_step
        self._running_history = dict()
        self._completed_avg_history = dict()
        if optimize_mode == 'maximize':
            self._high_better = True
        elif optimize_mode == 'minimize':
            self._high_better = False
        else:
            self._high_better = True
            logging.warning('unrecognized optimize_mode %s', optimize_mode)
        self.moving_avg = moving_avg
        self.proportion = proportion
        self.patience = patience
        self._patience_dict = dict()

    def _update_data(self, trial_job_id, trial_history):
        """update data

        Parameters
        ----------
        trial_job_id : int
            trial job id
        trial_history : list
            The history performance matrix of each trial
        """
        if trial_job_id not in self._running_history:
            self._running_history[trial_job_id] = []
            self._patience_dict[trial_job_id] = self.patience
        if not self.moving_avg:
            self._running_history[trial_job_id].extend(
                trial_history[len(self._running_history[trial_job_id]):])
        else:
            n = len(self._running_history[trial_job_id])
            if n:
                s = self._running_history[trial_job_id][-1] * n
            else:
                s = 0
            s += trial_history[-1]
            s /= (n + 1)
            self._running_history[trial_job_id].append(s)
            logging.info('job_id: %s, running_history: %s', trial_job_id,
                         self._running_history)

    def assess_trial(self, trial_job_id, trial_history):
        logging.info('trial access %s %s', trial_job_id, trial_history)

        # adapt to the trial_end for early_stop or user_cancel,nni.report_intermediate_result(0) at the begining
        curr_step = len(trial_history) - 1
        if len(trial_history) >= 2:
            scalar_trial_history = extract_scalar_history(trial_history[1:])
            logging.info('trial access: %s scalar_trial_history: %s',
                         trial_job_id, scalar_trial_history)
            self._update_data(trial_job_id, scalar_trial_history)

        if curr_step < self._start_step:
            return AssessResult.Good

        if self._high_better:
            best_history = max(scalar_trial_history)
            if self.patience is not None:
                logging.info('using patience to ealystop the trial ')
                if scalar_trial_history[-1] < best_history:
                    self._patience_dict[trial_job_id] -= 1
                    logging.info(
                        'use the maximize, the history %s < best_history %s, then patience is %s',
                        scalar_trial_history[-1], best_history,
                        self._patience_dict[trial_job_id])
                    if self._patience_dict[trial_job_id] <= 0:
                        logging.info(
                            'patience<=0, then we will stop the trial')
                        return AssessResult.Bad
                else:
                    self._patience_dict[trial_job_id] = self.patience

        else:
            best_history = min(scalar_trial_history)
            if self.patience is not None:
                if scalar_trial_history[-1] > best_history:
                    self._patience_dict[trial_job_id] -= 1
                    logging.info(
                        'use the maximize, the history %s < best_history %s, then patience is %s',
                        scalar_trial_history[-1], best_history,
                        self._patience_dict[trial_job_id])
                    if self._patience_dict[trial_job_id] <= 0:
                        logging.info(
                            'patience<=0, then we will stop the trial')
                        return AssessResult.Bad
                else:
                    self._patience_dict[trial_job_id] = self.patience

        avg_array = []

        for id_ in self._running_history:
            if id_ != trial_job_id:
                if len(self._running_history[id_]) >= curr_step:
                    avg_array.append(self._running_history[id_][curr_step - 1])
                else:
                    avg_array.append(self._running_history[id_][-1])

        if avg_array:
            avg_array.sort()
            if self._high_better:
                proportion_value = avg_array[int(
                    (len(avg_array) - 1) * self.proportion)]
                logging.info(
                    'avg_array: %s proportion_value: %s, trial best value %s',
                    avg_array, proportion_value, best_history)
                return AssessResult.Bad if best_history < proportion_value else AssessResult.Good
            else:
                proportion_value = avg_array[int(
                    len(avg_array) * self.proportion)]
                logging.info(
                    'avg_array: %s proportion_value: %s, trial best value %s',
                    avg_array, proportion_value, best_history)
                return AssessResult.Bad if best_history > proportion_value else AssessResult.Good
        else:
            return AssessResult.Good

    def trial_end(self, trial_job_id, success):
        logging.info('trial end')
