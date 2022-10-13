# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
import pathlib
import shutil

filepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

outfile_dir = os.path.join(filepath, 'exp_json')


def get_filepath(trial_id=None):
    pathlib.Path(outfile_dir).mkdir(parents=True, exist_ok=True)
    if trial_id:
        outfile = os.path.join(outfile_dir, str(trial_id) + '_mc.json')
    else:
        outfile = os.path.join(outfile_dir, 'mc.json')

    if not os.path.exists(outfile):
        with open(outfile, 'w') as f:
            json.dump({}, f)
    return outfile


def remove_filepath(trial_id=None):
    # delete the internal events file
    expdir = get_value(key='expdir', trial_id=trial_id)
    if expdir:
        print('removedir:', expdir)
        shutil.rmtree(expdir, ignore_errors=True)

    file = get_filepath(trial_id=trial_id)
    print('remove file', file)
    os.remove(file)


def set_value(key, value, trial_id=None):
    outfile = get_filepath(trial_id=trial_id)
    with open(outfile, 'r') as f:
        _global_dict = json.load(f)
    _global_dict[key] = value
    with open(outfile, 'w') as f:
        json.dump(_global_dict, f)


def get_value(key, defValue=None, trial_id=None):
    outfile = get_filepath(trial_id=trial_id)
    with open(outfile, 'r') as f:
        _global_dict = json.load(f)

    try:
        return _global_dict[key]
    except KeyError:
        return defValue
