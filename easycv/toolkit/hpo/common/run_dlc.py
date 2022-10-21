# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import logging
import os

import nni
from hpo_tools.core.metric.report_summary_metric import report_result
from hpo_tools.core.platform.dlc.dlc_utils import kill_job, run_multi_command
from hpo_tools.core.utils.config_utils import parse_ini
from hpo_tools.core.utils.json_utils import set_value
from hpo_tools.core.utils.path_utils import unique_path


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, help='config path', default='./config_oss.ini')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    try:
        args = get_params()
        logging.info('args: %s', args)

        config = parse_ini(args.config)

        cmd_config = config['cmd_config']
        logging.info('cmd_config: %s', cmd_config)

        oss_config = config.get('oss_config', None)

        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        trial_id = str(nni.get_trial_id())
        experment_id = str(nni.get_experiment_id())

        # update parameter
        for k, cmd in cmd_config.items():
            cmd = cmd.replace('${exp_id}', experment_id)
            cmd = cmd.replace('${trial_id}', trial_id)
            tuner_params_list = ''
            tuner_params_dict = ''
            for p, v in tuner_params.items():
                cmd = cmd.replace(p, str(v))
                tuner_params_list += p + ' ' + str(v) + ' '
                tuner_params_dict += p + '=' + str(v) + ' '
            cmd = cmd.replace('${tuner_params_list}', tuner_params_list)
            cmd = cmd.replace('${tuner_params_dict}', tuner_params_dict)
            cmd_config[k] = cmd

        # report metric
        metric_dict = config['metric_config']
        logging.info('metric dict: %s', metric_dict)
        metric_filepath = metric_dict['metric_filepath']
        metric_filepath = metric_filepath.replace('${exp_id}', experment_id)
        metric_filepath = metric_filepath.replace('${trial_id}', trial_id)
        metric_dict.pop('metric_filepath')

        if metric_filepath.startswith('oss'):
            dst_filepath = unique_path('../exp')
            set_value(
                'expdir', os.path.abspath(dst_filepath), trial_id=trial_id)
            ori_filepath = metric_filepath
        else:
            ori_filepath = None
            dst_filepath = metric_filepath

        report_result(
            ori_filepath,
            dst_filepath,
            metric_dict,
            trial_id,
            use_best=True,
            oss_config=oss_config)

        # for earlystop or user_canceled
        nni.report_intermediate_result(0)

        # run command
        run_multi_command(cmd_config, trial_id)

    except Exception:
        logging.exception('run begin error')
        exit(1)

    finally:
        # kill  instance
        kill_job(trial_id=trial_id)
        # for kill report result
        set_value(trial_id + '_exit', '1', trial_id=trial_id)
