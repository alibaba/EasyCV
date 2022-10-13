import os
import pathlib
import random
import string
from os.path import getsize, join
from threading import local

from torch.utils.tensorboard import SummaryWriter


def random_string_generator(str_size):
    return ''.join(
        random.choice(string.ascii_letters) for x in range(str_size))


tag = random_string_generator(12)
local_metric_dir = './tmp/' + tag
print('local_metric_dir:', os.path.abspath(local_metric_dir))
writer = SummaryWriter(log_dir=local_metric_dir)
step = 0


def getdirsize(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([getsize(join(root, name)) for name in files])
    return size


def write_summary_metric(remote_metric_dir, metric_dict={'auc': 1}):
    # this will be called in container,so don't use the trial_id to unique value and log_dir
    global step

    for k, v in metric_dict.items():
        writer.add_scalar(k, v, step)
    writer.flush()
    step = step + 1

    # use nas or oss, we use the local dir,because the oss write has some problems
    pathlib.Path(remote_metric_dir).mkdir(parents=True, exist_ok=True)
    os.system('cp -rf ' + local_metric_dir + '/. ' + remote_metric_dir)
