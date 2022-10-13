# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import logging
import re

import odps
from hpo_tools.core.utils.config_utils import try_parse
from hpo_tools.core.utils.json_utils import get_value, set_value
from odps.models import Instance

Command = collections.namedtuple('Command', ['name', 'project', 'parameters'])


def create_odps(trial_id):
    access_id = get_value('access_id', trial_id=trial_id)
    access_key = get_value('access_key', trial_id=trial_id)
    project = get_value('project', trial_id=trial_id)
    endpoint = get_value('endpoint', trial_id=trial_id)

    if access_id is None:
        return None

    o = odps.ODPS(
        access_id=access_id,
        secret_access_key=access_key,
        endpoint=endpoint,
        project=project)

    proj = o.get_project(project)
    if not o.exist_project(proj):
        raise ValueError('ODPS init failed, please check your project name.')
    return o


def parse_cmd_config(cmd_config):
    """When val='x', convert "'x'"->'x' when val="x",convert '"x"'->'x'."""
    print('cmd:', cmd_config)
    arrs = cmd_config.split('\n')
    params = {}
    pattern = r'''((?:[^-"']|"[^"]*"|'[^']*')+)'''
    for arr in arrs:
        # the value can be
        arr = re.split(pattern, arr)
        for i in range(len(arr)):
            if arr[i].startswith('name='):
                name = arr[i][5:].strip()
                print('name:', name)
            elif arr[i].startswith('name'):
                name = arr[i][4:].strip()
                print('name:', name)
            elif arr[i].startswith('project='):
                project = arr[i][8:].strip()
                print('project:', project)
            elif arr[i].startswith('project'):
                project = arr[i][7:].strip()
                print('project:', project)
            elif arr[i].startswith('D'):
                if arr[i].find('=') > 0:
                    # the arr[i] can be -DinputTablePartitions='pt=exclude_0'
                    k, val = arr[i].split('=', 1)
                    if val[0] == "'" and val[-1] == "'":
                        val = val[1:-1]
                    if val[0] == '"' and val[-1] == '"':
                        val = val[1:-1]
                    params[k[1:]] = try_parse(val)
    return Command(name=name, project=project, parameters=params)


def run_multi_command(cmd_config, trial_id=None):
    # parse command
    o = create_odps(trial_id=trial_id)
    for k, cmd in cmd_config.items():
        cmd = cmd.strip().strip('"').strip("'")
        if cmd.strip().lower().startswith('pai'):
            command = parse_cmd_config(cmd_config=cmd)

            run_single_xflow(command=command, trial_id=trial_id, o=o)
        else:
            if cmd.strip().lower().startswith(
                    'grant') or cmd.strip().lower().startswith('revoke'):
                instance = o.run_security_query(cmd)
            else:
                instance = o.run_sql(cmd)

            set_value(trial_id, str(instance.id), trial_id=trial_id)
            print('instance id:', instance.id)
            print(instance.get_logview_address())
            instance.wait_for_success()
            if not instance.is_successful():
                print('instance failed, exit')
                exit(1)


def kill_instance(trial_id):
    logging.info('kill instance')
    o = create_odps(trial_id=trial_id)
    instance = get_value(trial_id, trial_id=trial_id)
    if o and instance:
        instance_o = o.get_instance(instance)
        logging.info('instance.status %s', instance_o.status.value)
        if instance_o.status != Instance.Status.TERMINATED:
            logging.info('stop instance')
            o.stop_instance(instance)
            logging.info('stop instance success')


def run_single_xflow(command, trial_id=None, o=None):
    logging.info('command %s', command)
    if not o:
        o = create_odps(trial_id=trial_id)
    instance = o.run_xflow(
        xflow_name=command.name,
        xflow_project=command.project,
        parameters=command.parameters)
    for inst_name, inst in o.iter_xflow_sub_instances(instance):
        logging.info('inst name: %s', inst_name)
        logging.info(inst.get_logview_address())
        logging.info('instance id: %s', inst)
        set_value(trial_id, str(inst), trial_id=trial_id)

    if not instance.is_successful():
        print('instance failed, exit')
        exit(1)


def parse_easyrec_cmd_config(easyrec_cmd_config):
    """When val='x', convert "'x'"->'x' when val="x",convert '"x"'->'x'."""
    name = easyrec_cmd_config['-name']
    project = easyrec_cmd_config['-project']

    params = {}
    for k, val in easyrec_cmd_config.items():
        if k.startswith('-D'):
            if val[0] == "'" and val[-1] == "'":
                val = val[1:-1]
            if val[0] == '"' and val[-1] == '"':
                val = val[1:-1]
            params[k.replace('-D', '')] = try_parse(val)
    return Command(name=name, project=project, parameters=params)


def run_command(easyrec_cmd_config, trial_id=None):
    # parse command
    command = parse_easyrec_cmd_config(easyrec_cmd_config=easyrec_cmd_config)
    run_single_xflow(command, trial_id)
