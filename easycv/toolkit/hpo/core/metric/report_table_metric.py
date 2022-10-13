# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import time
from threading import Thread

import nni
from hpo_tools.core.platform.maxcompute.pyodps_utils import create_odps
from hpo_tools.core.utils.json_utils import get_value, remove_filepath


def get_result(
    query_sql,
    metric_dict={'auc': 1},
    trial_id=None,
    nni_report_final=True,
):
    o = create_odps(trial_id=trial_id)
    instance = o.execute_sql(query_sql)
    print(instance.get_logview_address())

    with instance.open_reader() as reader:
        pd_df = reader.to_pandas()

    print('query_res:', pd_df)

    if len(pd_df) == 0:
        return None

    temp = 0
    metric_report = {'default': 0}
    for key in metric_dict:
        temp += pd_df.loc[0, key] * metric_dict[key]
        metric_report[key] = pd_df.loc[0, key]
        print('key:', key, ' value:', pd_df.loc[0, key])
    metric_report['default'] = temp
    if nni_report_final:
        nni.report_final_result(metric_report)
    return metric_report


def report_result(query_sql_list,
                  metric_dict={'auc': 1},
                  trial_id=None,
                  final_mode='avg'):
    worker = Thread(
        target=load_loop,
        args=(query_sql_list, metric_dict, trial_id, final_mode))
    worker.start()


def load_loop(query_sql_list, metric_dict, trial_id, final_mode):
    best = {}
    sum = {}
    cnt = 0
    while True:
        result = get_result(
            query_sql=query_sql_list[cnt],
            metric_dict=metric_dict,
            trial_id=trial_id,
            nni_report_final=False,
        )
        if result:
            nni.report_intermediate_result(result)
            cnt += 1
            for key in result:
                sum[key] = sum.get(key, 0) + result[key]
            if best.get('default', 0) < result['default']:
                best[key] = result

        if cnt >= len(query_sql_list) or (trial_id and get_value(
                trial_id + '_exit', trial_id=trial_id) == '1'):
            # train end
            if cnt >= len(query_sql_list):
                if final_mode == 'avg':
                    for key in sum:
                        sum[key] = sum[key] / cnt
                    nni.report_final_result(sum)
                elif final_mode == 'best':
                    nni.report_final_result(best)
                else:
                    nni.report_final_result(result)
            # remove the json file
            remove_filepath(trial_id=trial_id)
            logging.info('the job end')
            break
        time.sleep(30)
