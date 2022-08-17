# conver to new
import torch

from easycv.models import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import (CONFIG_TEMPLATE_ZOO,
                                       mmcv_config_fromfile, rebuild_config)

if __name__ == '__main__':
    # cfg_path = '/apsara/xinyi.zxy/code/pr154/configs/detection/yolox/yolox_s_8xb16_300e_coco_asff_tood3.py'
    cfg_path = '/apsara/xinyi.zxy/code/pr154/configs/detection/yolox/yolox_s_8xb16_300e_coco_asff_reptood2.py'
    cfg = mmcv_config_fromfile(cfg_path)
    model = build_model(cfg.model)
    print(model)

    # ckpt_path = '/apsara/xinyi.zxy/pretrain/asff_tood3/epoch_300_new.pth'
    ckpt_path = '/apsara/xinyi.zxy/pretrain/ab_study/yolox_asff_reptood2.pth'
    model_ckpt = torch.load(ckpt_path)
    pretrain_model_state = model_ckpt['state_dict']

    # model.load_state_dict(pretrain_model_state)
    #
    # exit()

    model_state_dict = model.state_dict()  # ??model?key

    # of1 = open('new.txt','w')
    # for key in model_state_dict.keys():
    #     of1.writelines(key+'\n')
    #
    # of2 = open('pre.txt', 'w')
    # for key in pretrain_model_state.keys():
    #     of2.writelines(key + '\n')

    key_ori = [
        'backbone.stem', 'ERBlock_2.0', 'ERBlock_2.1.conv1',
        'ERBlock_2.1.block.0', 'ERBlock_3.0', 'ERBlock_3.1.conv1',
        'ERBlock_3.1.block.0', 'ERBlock_3.1.block.1', 'ERBlock_3.1.block.2',
        'ERBlock_4.0', 'ERBlock_4.1.conv1', 'ERBlock_4.1.block.0',
        'ERBlock_4.1.block.1', 'ERBlock_4.1.block.2', 'ERBlock_4.1.block.3',
        'ERBlock_4.1.block.4', 'ERBlock_5.0', 'ERBlock_5.1.conv1',
        'ERBlock_5.1.block.0', 'ERBlock_5.2'
    ]

    key_new = [
        'backbone.stage0', 'stage1.0', 'stage1.1', 'stage1.2', 'stage2.0',
        'stage2.1', 'stage2.2', 'stage2.3', 'stage2.4', 'stage3.0', 'stage3.1',
        'stage3.2', 'stage3.3', 'stage3.4', 'stage3.5', 'stage3.6', 'stage4.0',
        'stage4.1', 'stage4.2', 'stage4.3'
    ]

    print(len(key_ori) == len(key_new))

    for i, key in enumerate(pretrain_model_state):
        find = False
        for t_i, t_k in enumerate(key_ori):
            if t_k in key:
                find = True
                break
        if find:
            model_state_dict[key.replace(
                t_k, key_new[t_i])] = pretrain_model_state[key]
        else:
            model_state_dict[key] = pretrain_model_state[key]

    model.load_state_dict(model_state_dict)

    model_ckpt['state_dict'] = model_state_dict
    ckpt_path_new = '/apsara/xinyi.zxy/pretrain/ab_study/yolox_asff_reptood2_new.pth'
    torch.save(model_ckpt, ckpt_path_new)

    # load
    model_ckpt_new = torch.load(ckpt_path_new)
    pretrain_model_state_new = model_ckpt_new['state_dict']

    model.load_state_dict(pretrain_model_state_new)
    #
    # exit()
