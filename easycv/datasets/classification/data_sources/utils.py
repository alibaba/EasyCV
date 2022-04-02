import os

from mmcv.runner import get_dist_info
from tqdm import tqdm

from easycv.file import io


def split_listfile_byrank(list_file,
                          label_balance,
                          save_path='data/',
                          delimeter=' '):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return list_file

    lines = io.open(list_file).readlines()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if label_balance:
        label2files = {}
        for l in tqdm(lines):
            label = l.split(delimeter)[-1].strip()
            path = l.split(delimeter)[0]
            if label in label2files.keys():
                label2files[label].append(path)
            else:
                label2files[label] = [path]

        label_list = list(label2files.keys())
        length = int(len(label_list) / world_size) + 1

        for idx in tqdm(range(world_size)):
            with io.open(os.path.join(save_path, 'rank_%d.txt' % idx),
                         'w') as f:
                for j in range(idx * length, idx * length + length):
                    j_label = label_list[j % len(label_list)]
                    for p in label2files[j_label]:
                        f.write('%s %s\n' % (p, j_label))
    else:
        length = int(len(lines) / world_size) + 1
        for idx in range(world_size):
            with io.open(os.path.join(save_path, 'rank_%d.txt' % idx),
                         'w') as f:
                for j in range(idx * length, idx * length + length):
                    f.write(lines[j % len(lines)])

    return os.path.join(save_path, 'rank_%d.txt' % rank)
