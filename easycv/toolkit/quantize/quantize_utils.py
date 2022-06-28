# Copyright (c) Alibaba, Inc. and its affiliates.
import itertools

import mmcv
import MNN
import numpy as np
import torch
from mmcv.parallel import scatter_kwargs

from easycv.models.detection.utils import output_postprocess, postprocess
from easycv.models.detection.yolox.yolo_head import YOLOXHead


def quantize_config_check(device, backend, model_type=''):
    """
    check quantize config and return config setting
    """
    quantize_config = {}
    if device == 'cpu' and backend == 'PyTorch':
        quantize_config['device'] = 'cpu'
        quantize_config['backend'] = 'PyTorch'
    elif device == 'arm' and backend == 'MNN':
        quantize_config['device'] = 'arm'
        quantize_config['backend'] = 'MNN'
    else:
        raise ValueError(
            '{} device and {} backend is not supported. It can only be (cpu, PyTorch) or (arm, MNN).'
            .format(device, backend))

    if 'YOLOX' in model_type:
        quantize_config['non_traceable_module_class'] = [YOLOXHead]

    return quantize_config


def calib(model, data_loader):
    for cur_iter, data in enumerate(data_loader):
        # This is help to refine the quantized model's output, so no need to use all data.
        # More than 50 samples will not get better result, but will cost too much more time.
        if cur_iter > 50:
            return
        input_args, kwargs = scatter_kwargs(None, data,
                                            [torch.cuda.current_device()])
        with torch.no_grad():
            model(kwargs[0]['img'])


def replace_module(module,
                   replaced_module_type,
                   new_module_type,
                   replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.
    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.
    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type,
                                       new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


def model_output_shape(model_type, output_n):
    '''
        output shape should be setting for every models
    '''
    if model_type == 'YOLOX' or 'YOLOX_EDGE':
        output_shape = (output_n, 3549, 6)
    else:
        raise ValueError(
            'Model type {} is not supported. Please contact PAI engineers to Put forward your demand.'
            .format(model_type))

    return output_shape


def single_mnn_test(cfg, model_path, data_loader, imgs_per_gpu):
    '''
        MNN models test
    '''
    # build MNN interpreter, and get input tensor
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_all = interpreter.getSessionInputAll(session)
    name = list(input_all.keys())[0]
    input_image = interpreter.getSessionInput(session, name)
    correct = 0

    if hasattr(data_loader, 'dataset'):  # normal dataloader
        data_len = len(data_loader.dataset)
    else:
        data_len = len(data_loader) * data_loader.batch_size

    prog_bar = mmcv.ProgressBar(data_len)
    results = {}
    for i, data in enumerate(data_loader):
        # use scatter_kwargs to unpack DataContainer data for raw torch.nn.module
        input_args, kwargs = scatter_kwargs(None, data, [-1])
        kwargs[0]['img'] = kwargs[0]['img'].squeeze(dim=0)
        images = kwargs[0]['img']
        img_meta = kwargs[0]['img_metas']
        images = images.cpu().numpy()

        # transfer ndarray to MNN.tensor
        image_mnn_tensor = MNN.Tensor(images.shape, MNN.Halide_Type_Float,
                                      images, MNN.Tensor_DimensionType_Caffe)

        input_image.copyFromHostTensor(image_mnn_tensor)

        # run MNN Session
        interpreter.runSession(session)

        # get MNN session output
        output_tensor = interpreter.getSessionOutputAll(session)

        # get output shape
        output_shape = model_output_shape(cfg.model.type, imgs_per_gpu)

        # transfor MNN's tensor to PyTorch's tensor
        tmp_output = MNN.Tensor(output_shape, MNN.Halide_Type_Float,
                                np.ones(list(output_shape)).astype(np.float32),
                                MNN.Tensor_DimensionType_Caffe)
        output_name = list(output_tensor.keys())[0]
        output = output_tensor[output_name]
        output.copyToHostTensor(tmp_output)
        output = tmp_output.getData()
        output = torch.tensor(output).view(output_shape)

        output = postprocess(output, cfg.model.num_classes,
                             cfg.model.test_conf, cfg.model.nms_thre)
        output = output_postprocess(output, img_meta)

        for k, v in output.items():
            if k not in results:
                results[k] = []
            results[k].append(v)

        if 'img_metas' in data:
            batch_size = len(data['img_metas'].data[0])
        else:
            batch_size = data['img'].size(0)

        for _ in range(batch_size):
            prog_bar.update()

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
