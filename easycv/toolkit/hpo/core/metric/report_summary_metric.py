# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import time
from pathlib import Path
from threading import Thread

import nni
from hpo_tools.core.utils.file_io import copy_dir
from hpo_tools.core.utils.json_utils import (get_value, remove_filepath,
                                             set_value)
from tensorflow.python.summary import summary_iterator


def judge_key(metric_dict, event_res):
    for key in metric_dict:
        if key not in event_res.keys():
            return False
    return True


def _get_best_by_step(metric_dict={'auc': 1},
                      trial_id=None,
                      use_best=False,
                      nni_report=True,
                      event_eval_result={},
                      best_eval_result={},
                      best_event=None):
    report_step = -1
    if trial_id:
        report_step = get_value(
            trial_id + '_report_step', -1, trial_id=trial_id)
    if len(event_eval_result) >= 2 and judge_key(metric_dict,
                                                 event_eval_result):
        temp = 0
        metric_report = {'default': 0}
        for key in metric_dict:
            temp += metric_dict[key] * event_eval_result[key]
            metric_report[key] = event_eval_result[key]
        metric_report['default'] = temp

        if use_best:
            if best_eval_result.get('default', 0) < temp:
                best_eval_result = metric_report
                best_event = event_eval_result
        else:  # use final result
            best_eval_result = metric_report
            best_event = event_eval_result

        if event_eval_result['global_step'] > report_step and nni_report:
            nni.report_intermediate_result(metric_report)
            if trial_id:
                set_value(
                    trial_id + '_report_step',
                    event_eval_result['global_step'],
                    trial_id=trial_id)
            logging.info('event_eval_result: %s, temp metric: %s',
                         event_eval_result, metric_report)
        return best_eval_result, best_event
    return {}, None


def _get_best_eval_result(event_files,
                          metric_dict={'auc': 1},
                          trial_id=None,
                          use_best=False,
                          nni_report=True,
                          nni_report_final=False,
                          dst_filepath=None):
    if not event_files:
        return None

    best_eval_result = {}
    best_event = None

    path1 = Path(os.path.abspath(dst_filepath))

    try:
        global_step = None
        event_eval_result = {}
        for event_file in path1.rglob(event_files):
            print('event_file:', event_file)
            event_final = None
            for event in summary_iterator.summary_iterator(str(event_file)):
                if event.HasField('summary'):
                    # torch event: {'global_step': 3286, 'auc': 0.26267778873443604}
                    #               {'global_step': 3286, 'auc2': 0.26267778873443604}
                    # tf event: {'global_step': 3286, 'auc': 0.26267778873443604, 'auc2':0.26267778873443604}
                    if global_step is not None and event.step != global_step:
                        best_eval_result, best_event = _get_best_by_step(
                            metric_dict=metric_dict,
                            trial_id=trial_id,
                            use_best=use_best,
                            nni_report=nni_report,
                            event_eval_result=event_eval_result,
                            best_eval_result=best_eval_result,
                            best_event=best_event)
                        event_eval_result = {}
                    global_step = event.step
                    event_eval_result['global_step'] = event.step
                    for value in event.summary.value:
                        if value.HasField('simple_value'):
                            event_eval_result[value.tag] = value.simple_value
                    event_final = event
            if event_final:
                best_eval_result, best_event = _get_best_by_step(
                    metric_dict=metric_dict,
                    trial_id=trial_id,
                    use_best=use_best,
                    nni_report=nni_report,
                    event_eval_result=event_eval_result,
                    best_eval_result=best_eval_result,
                    best_event=best_event)
    except Exception:
        logging.exception('the events is not ok,read the events error')

    if best_eval_result and nni_report and nni_report_final:
        nni.report_final_result(best_eval_result)

    return best_eval_result, best_event


def get_result(filepath,
               dst_filepath,
               metric_dict={'auc': 1},
               trial_id=None,
               oss_config=None,
               nni_report=True,
               use_best=False):
    if filepath:
        copy_dir(filepath, dst_filepath, oss_config)
    event_file_pattern = '*.tfevents.*'
    logging.info('event_file: %s', event_file_pattern)
    best_eval_result, best_event = _get_best_eval_result(
        event_file_pattern,
        metric_dict=metric_dict,
        trial_id=trial_id,
        nni_report=nni_report,
        use_best=use_best,
        dst_filepath=dst_filepath)
    logging.info('best_metric: %s', best_eval_result)
    logging.info('best_event: %s', best_event)
    return best_eval_result, best_event


def report_result(filepath,
                  dst_filepath,
                  metric_dict,
                  trial_id=None,
                  oss_config=None,
                  nni_report=True,
                  use_best=False):
    worker = Thread(
        target=load_loop,
        args=(filepath, dst_filepath, metric_dict, trial_id, oss_config,
              nni_report, use_best))
    worker.start()


def load_loop(filepath, dst_filepath, metric_dict, trial_id, oss_config,
              nni_report, use_best):
    while True:
        print('******new get_result start******')
        best_eval_result, best_event = get_result(
            filepath,
            dst_filepath,
            metric_dict=metric_dict,
            trial_id=trial_id,
            oss_config=oss_config,
            nni_report=nni_report,
            use_best=use_best)
        # train end normaly
        if trial_id and get_value(
                trial_id + '_exit', trial_id=trial_id) == '1':
            if best_eval_result.get('default', None):
                nni.report_final_result(best_eval_result)
            # remove the json file
            remove_filepath(trial_id=trial_id)
            logging.info('the job end')
            break
        time.sleep(30)
