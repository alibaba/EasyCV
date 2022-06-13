# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools
import os.path as osp
import pickle
import tempfile
import time
from io import BytesIO

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import (MMDataParallel, MMDistributedDataParallel,
                           scatter_kwargs)
from mmcv.runner import get_dist_info

from easycv.file import io


def single_cpu_test(model,
                    data_loader,
                    mode='test',
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    **kwargs):

    model.eval()
    if hasattr(data_loader, 'dataset'):  # normal dataloader
        data_len = len(data_loader.dataset)
    else:
        data_len = len(data_loader) * data_loader.batch_size

    prog_bar = mmcv.ProgressBar(data_len)
    results = {}

    for i, data in enumerate(data_loader):
        # use scatter_kwargs to unpack DataContainer data for raw torch.nn.module
        input_args, kwargs = scatter_kwargs(None, data, [-1])
        with torch.no_grad():
            result = model(**kwargs[0], mode=mode)

        for k, v in result.items():
            if k not in results:
                results[k] = []
            results[k].append(v)

        if 'img_metas' in data:
            batch_size = len(data['img_metas'].data[0])
        else:
            batch_size = data['img'].size(0)

        for _ in range(batch_size):
            prog_bar.update()

    # new line for prog_bar
    print()
    for k, v in results.items():
        if len(v) == 0:
            raise ValueError(f'empty result for {k}')

        if isinstance(v[0], torch.Tensor):
            results[k] = torch.cat(v, 0)
        elif isinstance(v[0], (list, np.ndarray)):
            results[k] = list(itertools.chain.from_iterable(v))
        else:
            raise ValueError(
                f'value of batch prediction dict should only be tensor or list, {k} type is {v[0]}'
            )

    return results


def single_gpu_test(model, data_loader, mode='test', use_fp16=False, **kwargs):
    """Test model with single.

    This method tests model with single

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        model (str): mode for model to forward
        use_fp16:  Use fp16 inference

    Returns:
        list: The prediction results.
    """

    if use_fp16:
        device = next(model.parameters()).device
        assert device.type == 'cuda', 'fp16 can only be used in gpu, model is placed on cpu'
        model.half()

    model.eval()
    if hasattr(data_loader, 'dataset'):  # normal dataloader
        data_len = len(data_loader.dataset)
    else:
        data_len = len(data_loader) * data_loader.batch_size

    prog_bar = mmcv.ProgressBar(data_len)
    results = {}
    for i, data in enumerate(data_loader):
        # use scatter_kwargs to unpack DataContainer data for raw torch.nn.module
        if not isinstance(model, MMDistributedDataParallel) and not isinstance(
                model, MMDataParallel):
            input_args, kwargs = scatter_kwargs(None, data,
                                                [torch.cuda.current_device()])
            with torch.no_grad():
                result = model(**kwargs[0], mode=mode)
        else:
            with torch.no_grad():
                result = model(**data, mode=mode)

        for k, v in result.items():
            if k not in results:
                results[k] = []
            results[k].append(v)

        if 'img_metas' in data:
            if isinstance(data['img_metas'], list):
                batch_size = len(data['img_metas'][0].data[0])
            else:
                batch_size = len(data['img_metas'].data[0])

        else:
            if isinstance(data['img'], list):
                batch_size = data['img'][0].size(0)
            else:
                batch_size = data['img'].size(0)

        for _ in range(batch_size):
            prog_bar.update()

    # new line for prog_bar
    print()
    for k, v in results.items():
        if len(v) == 0:
            raise ValueError(f'empty result for {k}')

        if isinstance(v[0], torch.Tensor):
            results[k] = torch.cat(v, 0)
        elif isinstance(v[0], (list, np.ndarray)):
            results[k] = list(itertools.chain.from_iterable(v))
        else:
            raise ValueError(
                f'value of batch prediction dict should only be tensor or list, {k} type is {v[0]}'
            )

    return results


def multi_gpu_test(model,
                   data_loader,
                   mode='test',
                   tmpdir=None,
                   gpu_collect=False,
                   use_fp16=False,
                   **kwargs):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        model (str): mode for model to forward
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        use_fp16:  Use fp16 inference

    Returns:
        list: The prediction results.
    """
    if use_fp16:
        device = next(model.parameters()).device
        assert device.type == 'cuda', 'fp16 can only be used in gpu, model is placed on cpu'
        model.half()

    model.eval()
    results = {}
    rank, world_size = get_dist_info()

    if hasattr(data_loader, 'dataset'):  # normal dataloader
        data_len = len(data_loader.dataset)
    else:
        data_len = len(data_loader) * data_loader.batch_size * world_size

    if rank == 0:
        prog_bar = mmcv.ProgressBar(data_len)
    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(**data, mode=mode)
            # # encode mask results
            # if isinstance(result, tuple):
            #     bbox_results, mask_results = result
            #     encoded_mask_results = encode_mask_results(mask_results)
            #     result = bbox_results, encoded_mask_results

        for k, v in result.items():
            if k not in results:
                results[k] = []

            results[k].append(v)

        if rank == 0:
            if 'img_metas' in data:
                batch_size = len(data['img_metas'].data[0])
            else:
                batch_size = data['img'].size(0)
            # on DLC test bar while print too much log
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # new line for prog_bar
    # print("gpu_collect", gpu_collect)

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, data_len)
    else:
        results = collect_results_cpu(results, data_len, tmpdir)

    if rank == 0:
        for k, v in results.items():
            if len(v) == 0:
                raise ValueError(f'empty result for {k}')
            if isinstance(v[0], torch.Tensor):
                results[k] = torch.cat(v, 0)
            elif isinstance(v[0], list):
                results[k] = list(itertools.chain.from_iterable(v))
            else:
                raise ValueError(
                    f'value of batch prediction dict should only be tensor or list, {k} type is {v[0]}'
                )

    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    with io.open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as ofile:
        mmcv.dump(result_part, ofile, file_format='pkl')
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_dict = {}
        for i in range(world_size):
            part_file = io.open(osp.join(tmpdir, f'part_{i}.pkl'), 'rb')
            for k, v in mmcv.load(part_file, file_format='pkl').items():
                if k not in part_dict:
                    part_dict[k] = []
                part_dict[k].extend(v)
        # # sort the results
        # ordered_results = []
        # for res in zip(*part_list):
        #     ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = {k: v[:size] for k, v in part_dict.items()}
        # remove tmp dir
        io.rmtree(tmpdir)
        return ordered_results


def serialize_tensor(tensor_collection):
    buffer = BytesIO()
    torch.save(tensor_collection, buffer)
    return buffer


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle

    # part_tensor = torch.tensor(
    #     bytearray(serialize_tensor(result_part)), dtype=torch.uint8, device='cuda')
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_dict = {}
        for recv, shape in zip(part_recv_list, shape_list):
            result_part = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            for k, v in result_part.items():
                if k not in part_dict:
                    part_dict[k] = []
                part_dict[k].extend(v)

        # # sort the results
        # ordered_results = []
        # for res in zip(*part_list):
        #     ordered_results.extend(list(res))
        # the dataloader may pad some samples
        # ordered_results = ordered_results[:size]
        ordered_results = {k: v[:size] for k, v in part_dict.items()}
        return ordered_results
