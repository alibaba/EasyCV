# Copyright (c) Alibaba, Inc. and its affiliates.
try:
    from nni.algorithms.compression.pytorch.pruning import AGPPrunerV2
except ImportError:
    raise ImportError(
        'Please read docs and run "pip install https://pai-nni.oss-cn-zhangjiakou.aliyuncs.com/release/2.5/pai_nni-2.5-py3-none-manylinux1_x86_64.whl" '
        'to install pai_nni')


def get_prune_layer(model):
    if model == 'YOLOX_EDGE':
        backbone_name = 'backbone'

        prune_layer_names = []

        dark_block_num = {
            2: 3,
            3: 9,
            4: 9,
            5: 3,
            'C3_p4': 3,
            'C3_p3': 3,
            'C3_n3': 3,
            'C3_n4': 3,
        }

        for dark_id in [2, 3, 4]:
            prune_layer_names.append(
                f'{backbone_name}.backbone.dark{dark_id}.0.pconv.conv')
            for m_id in range(dark_block_num[dark_id]):
                prune_layer_names.append(
                    f'{backbone_name}.backbone.dark{dark_id}.1.m.{m_id}.conv1.conv'
                )

            prune_layer_names.append(
                f'{backbone_name}.backbone.dark{dark_id}.1.conv3.conv')

        for dark_id in [5]:
            prune_layer_names.append(
                f'{backbone_name}.backbone.dark{dark_id}.0.pconv.conv')
            prune_layer_names.append(
                f'{backbone_name}.backbone.dark{dark_id}.1.conv2.conv')
            prune_layer_names.append(
                f'{backbone_name}.backbone.dark{dark_id}.2.conv1.conv')
            for m_id in range(dark_block_num[dark_id]):
                prune_layer_names.append(
                    f'{backbone_name}.backbone.dark{dark_id}.2.m.{m_id}.conv1.conv'
                )
                prune_layer_names.append(
                    f'{backbone_name}.backbone.dark{dark_id}.2.m.{m_id}.conv2.pconv.conv'
                )

            prune_layer_names.append(
                f'{backbone_name}.backbone.dark{dark_id}.2.conv3.conv')

        for dark_id in ['C3_p4', 'C3_p3', 'C3_n3', 'C3_n4']:
            prune_layer_names.append(f'{backbone_name}.{dark_id}.conv1.conv')
            for m_id in range(dark_block_num[dark_id]):
                prune_layer_names.append(
                    f'{backbone_name}.{dark_id}.m.{m_id}.conv1.conv')
                prune_layer_names.append(
                    f'{backbone_name}.{dark_id}.m.{m_id}.conv2.pconv.conv')

            prune_layer_names.append(f'{backbone_name}.{dark_id}.conv2.conv')
            prune_layer_names.append(f'{backbone_name}.{dark_id}.conv3.conv')
        return prune_layer_names
    else:
        # default layer
        return ['Conv2d']


def load_pruner(model,
                pruner_config,
                optimizer,
                pruning_class='AGP',
                pruning_algorithm='taylorfo'):
    # default pruning class is AGPPrunerV2
    if pruning_class == 'AGP':
        # default pruning algorithm is taylorfo
        pruner = AGPPrunerV2(
            model=model,
            config_list=pruner_config,
            optimizer=optimizer,
            pruning_algorithm=pruning_algorithm)
    else:
        raise Exception(
            'pruning class {} is not supported'.format(pruning_class))

    return pruner
