import argparse
import numpy as np
from torchvision.models import resnet50
import torch
from torch.backends import cudnn
import tqdm

from easycv.models import build_model
from easycv.utils.config_tools import mmcv_config_fromfile
cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(
        description='EasyCV model memory and inference_time test')
    parser.add_argument('config', help='test config file path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv_config_fromfile(args.config)

    device = 'cuda:7'
    model = build_model(cfg.model).to(device)
    repetitions = 300

    dummy_input = torch.rand(1, 3, 224, 224).to(device)

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model.forward_test(dummy_input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()


    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model.forward_test(dummy_input)
            ender.record()
            torch.cuda.synchronize() # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = timings.sum()/repetitions
    print(torch.cuda.memory_summary(device))
    print('\navg={}\n'.format(avg))

if __name__ == '__main__':
    main()