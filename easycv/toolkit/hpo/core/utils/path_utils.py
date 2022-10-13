import os

import nni


def unique_path(path):
    # for tag
    experment_id = str(nni.get_experiment_id())
    trial_id = str(nni.get_trial_id())
    return os.path.join(path, experment_id + '_' + trial_id)
