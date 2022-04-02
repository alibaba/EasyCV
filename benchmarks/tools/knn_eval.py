# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

args = argparse.ArgumentParser(description='Process some integers.')
args.add_argument(
    'feature_dir',
    type=str,
    help='feature root dir',
    nargs='?',
    default='work_dirs')
args.add_argument(
    'train_path',
    type=str,
    help='train_path to match train feature npy',
    nargs='?',
    default='train_*')
args.add_argument(
    'val_path',
    type=str,
    help='train_path to match train feature npy',
    nargs='?',
    default='val_*')
args.add_argument(
    '--nb_knn',
    default=[10, 20, 100, 200],
    nargs='+',
    type=int,
    help='Number of NN to use. 20 is usually working the best.')
args.add_argument(
    '--temperature',
    default=0.07,
    type=float,
    help='Temperature used in the voting coefficient')


def knn_classifier(train_features,
                   train_labels,
                   test_features,
                   test_labels,
                   k,
                   T,
                   num_classes=1000):

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx:min((idx +
                                          imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx:min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)

        retrieved_neighbors = torch.gather(candidates, 1, indices)
        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(
            5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


if __name__ == '__main__':
    args = args.parse_args()

    train_list = glob('%s/%sfeat1.npy' % (args.feature_dir, args.train_path))
    val_list = glob('%s/%sfeat1.npy' % (args.feature_dir, args.val_path))

    train_features = []
    train_labels = []
    val_features = []
    val_labels = []

    for i in tqdm(train_list):
        label_npy = i.replace('feat1', 'label')
        train_features.append(np.load(i))
        train_labels.append(np.load(label_npy))

    train_features = torch.tensor(np.vstack(train_features)).cuda()
    train_labels = torch.tensor(np.hstack(train_labels)).long().cuda()
    print(train_features.shape)
    print(train_labels.shape)

    for i in tqdm(val_list):
        label_npy = i.replace('feat1', 'label')
        val_features.append(np.load(i))
        val_labels.append(np.load(label_npy))

    val_features = torch.tensor(np.vstack(val_features)).cuda()
    val_labels = torch.tensor(np.hstack(val_labels)).long().cuda()

    print(val_features.shape)
    print(val_labels.shape)

    train_features = nn.functional.normalize(train_features, dim=1, p=2).cuda()
    val_features = nn.functional.normalize(val_features, dim=1, p=2).cuda()

    for k in args.nb_knn:

        print(
            k,
            knn_classifier(train_features, train_labels, val_features,
                           val_labels, k, args.temperature))
