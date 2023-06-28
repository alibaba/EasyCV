# Copyright (c) Alibaba, Inc. and its affiliates.

# isort:skip_file
import argparse
import copy
import functools
import glob
import inspect
import logging
import os
import threading
import traceback
import torch
from mmcv import DictAction
from easycv.file import io

try:
    import easy_predict
except ModuleNotFoundError:
    print('please install easy_predict first using following instruction')
    print(
        'pip install https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/release/easy_predict-0.4.2-py2.py3-none-any.whl'
    )
    exit()

from easy_predict import (Base64DecodeProcess, DataFields,
                          DefaultResultFormatProcess, DownloadProcess,
                          FileReadProcess, FileWriteProcess, Process,
                          ProcessExecutor, ResultGatherProcess,
                          TableReadProcess, TableWriteProcess)
from mmcv.runner import init_dist

from easycv.utils.dist_utils import get_dist_info
from easycv.utils.logger import get_root_logger


def define_args():
    parser = argparse.ArgumentParser('easycv prediction')
    parser.add_argument(
        '--model_type',
        default='',
        help='model type, classifier/detector/segmentor/yolox')
    parser.add_argument('--model_path', default='', help='path to model')
    parser.add_argument(
        '--model_config',
        default='',
        help='model config str, predictor v1 param')

    # oss input output
    parser.add_argument(
        '--input_file',
        default='',
        help='filelist for images, eash line is a oss path or a local path')
    parser.add_argument(
        '--output_file',
        default='',
        help='oss file or local file to save predict info')
    parser.add_argument(
        '--output_dir',
        default='',
        help='output_directory to save image and video results')
    parser.add_argument(
        '--oss_prefix',
        default='',
        help='oss_prefix will be replaced with local_prefix in input_file')
    parser.add_argument(
        '--local_prefix',
        default='',
        help='oss_prefix will be replaced with local_prefix in input_file')

    # table input output
    parser.add_argument('--input_table', default='', help='input table name')
    parser.add_argument('--output_table', default='', help='output table name')
    parser.add_argument('--image_col', default='', help='input image column')
    parser.add_argument(
        '--reserved_columns',
        default='',
        help=
        'columns from input table to be saved to output table, comma seperated'
    )
    parser.add_argument(
        '--result_column',
        default='',
        help='result columns to be saved to output table, comma seperated')
    parser.add_argument(
        '--odps_config',
        default='./odps.config',
        help='path to your odps config file')
    parser.add_argument(
        '--image_type', default='url', help='image data type, url or base64')

    # common args
    parser.add_argument(
        '--queue_size',
        type=int,
        default=1024,
        help='length of queues used for each process')
    parser.add_argument(
        '--predict_thread_num',
        type=int,
        default=1,
        help='number of threads used for prediction')
    parser.add_argument(
        '--preprocess_thread_num',
        type=int,
        default=1,
        help='number of threads used for preprocessing and downloading')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='batch size used for prediction')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        type=str,
        choices=[None, 'pytorch'],
        help='if assigned pytorch, should be used in gpu environment')
    parser.add_argument(
        '--oss_io_config',
        nargs='+',
        action=DictAction,
        help='designer needs a oss of config to access the data')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


class PredictorProcess(Process):

    def __init__(self,
                 predict_fn,
                 batch_size,
                 thread_num,
                 local_rank=0,
                 input_queue=None,
                 output_queue=None):
        job_name = 'Predictor'
        if torch.cuda.is_available():
            thread_init_fn = functools.partial(torch.cuda.set_device,
                                               local_rank)
        else:
            thread_init_fn = None
        super(PredictorProcess, self).__init__(
            job_name,
            thread_num,
            input_queue,
            output_queue,
            batch_size=batch_size,
            thread_init_fn=thread_init_fn)
        self.predict_fn = predict_fn
        self.data_lock = threading.Lock()
        self.all_data_failed = True
        self.input_empty = True
        self.local_rank = 0

    def process(self, input_batch):
        """
    Read a batch of image from input_queue and predict

    Args:
      input_batch: a batch of input data
    Returns:
      output_queue:  unstak batch, push input data and prediction result into queue
    """
        valid_input = []
        valid_indices = []
        valid_frame_ids = []
        if self.batch_size == 1:
            input_batch = [input_batch]
        if self.input_empty and len(input_batch) > 0:
            self.data_lock.acquire()
            self.input_empty = False
            self.data_lock.release()

        output_data_list = input_batch
        for out in output_data_list:
            out[DataFields.prediction_result] = None

        for idx, input_data in enumerate(input_batch):
            if DataFields.image in input_data \
                and input_data[DataFields.image] is not None:
                valid_input.append(input_data[DataFields.image])
                valid_indices.append(idx)

        if len(valid_input) > 0:
            try:
                # flatten video_clip to images, use image predictor to predict
                # then regroup the result to a list for one video_clip
                output_list = self.predict_fn(valid_input)

                if len(output_list) > 0:
                    assert isinstance(output_list[0], dict), \
                        'the element in predictor output must be a dict'

                if self.all_data_failed:
                    self.data_lock.acquire()
                    self.all_data_failed = False
                    self.data_lock.release()

            except Exception:
                logging.error(traceback.format_exc())
                output_list = [None for i in range(len(valid_input))]

            for idx, result_dict in zip(valid_indices, output_list):
                output_data = output_data_list[idx]
                output_data[DataFields.prediction_result] = result_dict
                if result_dict is None:
                    output_data[DataFields.error_msg] = 'prediction error'

                output_data_list[idx] = output_data

        for output_data in output_data_list:
            self.put(output_data)

    def destroy(self):
        if not self.input_empty and self.all_data_failed:
            raise RuntimeError(
                'failed to predict all the input data, please see exception throwed above in the log'
            )


def create_yolox_predictor_kwargs(model_dir):
    jit_models = glob.glob('%s/**/*.jit' % model_dir, recursive=True)
    raw_models = glob.glob('%s/**/*.pt' % model_dir, recursive=True)
    if len(jit_models) > 0:
        assert len(
            jit_models
        ) == 1, f'more than one jit script model files is found in {model_dir}'
        config_path = jit_models[0] + '.config.json'
        if not os.path.exists(config_path):
            raise ValueError(
                f'Not find config json file {config_path} for inference with jit script model'
            )
        return {'model_path': jit_models[0], 'config_file': config_path}
    else:
        assert len(raw_models) > 0, f'export model not found in {model_dir}'
        assert len(raw_models
                   ) == 1, f'more than one model files is found in {model_dir}'
        return {'model_path': raw_models[0]}


def create_default_predictor_kwargs(model_dir):
    model_path = glob.glob('%s/**/*.pt*' % model_dir, recursive=True)
    assert len(model_path) > 0, f'model not found in {model_dir}'
    assert len(
        model_path) == 1, f'more than one model file is found {model_path}'
    model_path = model_path[0]
    logging.info(f'model found: {model_path}')

    config_path = glob.glob('%s/**/*.py' % model_dir, recursive=True)
    if len(config_path) == 0:
        config_path = None
    else:
        assert len(config_path
                   ) == 1, f'more than one config file is found {config_path}'
        config_path = config_path[0]
        logging.info(f'config found: {config_path}')
    if config_path:
        return {'model_path': model_path, 'config_file': config_path}
    else:
        return {'model_path': model_path, 'config_file': None}


def create_predictor_kwargs(model_type, model_dir):
    if model_type == 'YoloXPredictor':
        return create_yolox_predictor_kwargs(model_dir)
    else:
        return create_default_predictor_kwargs(model_dir)


def init_predictor(args):
    model_type = args.model_type
    model_path = args.model_path
    batch_size = args.batch_size
    from easycv.predictors.builder import build_predictor

    ori_model_path = model_path
    if os.path.isdir(ori_model_path):
        predictor_kwargs = create_predictor_kwargs(model_type, ori_model_path)
    else:
        predictor_kwargs = {'model_path': ori_model_path}

    predictor_cfg = dict(type=model_type, **predictor_kwargs)
    if args.model_config != '':
        predictor_cfg['model_config'] = args.model_config
    predictor = build_predictor(predictor_cfg)
    return predictor


def replace_oss_with_local_path(ori_file, dst_file, bucket_prefix,
                                local_prefix):
    bucket_prefix = bucket_prefix.rstrip('/') + '/'
    local_prefix = local_prefix.rstrip('/') + '/'
    with io.open(ori_file, 'r') as infile:
        with open(dst_file, 'w') as ofile:
            for l in infile:
                if l.startswith('oss://'):
                    l = l.replace(bucket_prefix, local_prefix)
                ofile.write(l)


def build_and_run_file_io(args):
    # distribute info
    rank, world_size = get_dist_info()
    worker_id = rank

    # check oss_config and init oss io
    if args.oss_io_config is not None:
        io.access_oss(**args.oss_io_config)

    # acquire the temporary save path
    if args.output_file:
        io.makedirs(os.path.dirname(args.output_file))
        input_oss_file_new_host = os.path.join(
            os.path.dirname(args.output_file),
            os.path.basename(args.input_file + '.tmp%d' % worker_id))
        replace_oss_with_local_path(args.input_file, input_oss_file_new_host,
                                    args.oss_prefix, args.local_prefix)
    else:
        io.makedirs(args.output_dir)
        input_oss_file_new_host = os.path.join(
            args.output_dir,
            os.path.basename(args.input_file + '.tmp%d' % worker_id))
        replace_oss_with_local_path(args.input_file, input_oss_file_new_host,
                                    args.oss_prefix, args.local_prefix)

    args.input_file = input_oss_file_new_host
    num_worker = world_size
    print(f'worker num {num_worker}')
    print(f'worker_id {worker_id}')
    batch_size = args.batch_size
    print(f'Local rank {args.local_rank}')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    predictor = init_predictor(args)
    predict_fn = predictor.__call__ if hasattr(
        predictor, '__call__') else predictor.predict
    # create proc executor
    proc_exec = ProcessExecutor(args.queue_size)

    # create oss read process to read file path from filelist
    proc_exec.add(
        FileReadProcess(
            args.input_file,
            slice_id=worker_id,
            slice_count=num_worker,
            output_queue=proc_exec.get_output_queue()))

    # download and decode image data
    proc_exec.add(
        DownloadProcess(
            thread_num=args.predict_thread_num,
            input_queue=proc_exec.get_input_queue(),
            output_queue=proc_exec.get_output_queue(),
            is_video_url=False))

    # transform image data
    proc_exec.add(
        PredictorProcess(
            input_queue=proc_exec.get_input_queue(),
            output_queue=proc_exec.get_output_queue(),
            predict_fn=predict_fn,
            batch_size=batch_size,
            local_rank=args.local_rank,
            thread_num=args.predict_thread_num))

    proc_exec.add(
        DefaultResultFormatProcess(
            input_queue=proc_exec.get_input_queue(),
            output_queue=proc_exec.get_output_queue()))

    # Gather result to different dict of different type
    proc_exec.add(
        ResultGatherProcess(
            output_type_dict={},
            input_queue=proc_exec.get_input_queue(),
            output_queue=proc_exec.get_output_queue()))

    # Write result
    proc_exec.add(
        FileWriteProcess(
            output_file=args.output_file,
            output_dir=args.output_dir,
            slice_id=worker_id,
            slice_count=num_worker,
            input_queue=proc_exec.get_input_queue()))

    proc_exec.run()
    proc_exec.wait()


def build_and_run_table_io(args):
    os.environ['ODPS_CONFIG_FILE_PATH'] = args.odps_config

    rank, world_size = get_dist_info()
    worker_id = rank
    num_worker = world_size
    print(f'worker num {num_worker}')
    print(f'worker_id {worker_id}')

    batch_size = args.batch_size
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    predictor = init_predictor(args)
    predict_fn = predictor.__call__ if hasattr(
        predictor, '__call__') else predictor.predict
    # batch size should be less than the total number of data in input table
    table_read_batch_size = 1
    table_read_thread_num = 4

    # create proc executor
    proc_exec = ProcessExecutor(args.queue_size)

    # create oss read process to read file path from filelist
    selected_cols = list(
        set(args.image_col.split(',') + args.reserved_columns.split(',')))
    if args.image_col not in selected_cols:
        selected_cols.append(args.image_col)
    image_col_idx = selected_cols.index(args.image_col)
    proc_exec.add(
        TableReadProcess(
            args.input_table,
            selected_cols=selected_cols,
            slice_id=worker_id,
            slice_count=num_worker,
            output_queue=proc_exec.get_output_queue(),
            image_col_idx=image_col_idx,
            image_type=args.image_type,
            batch_size=table_read_batch_size,
            num_threads=table_read_thread_num))

    if args.image_type == 'base64':
        base64_thread_num = args.preprocess_thread_num
        proc_exec.add(
            Base64DecodeProcess(
                thread_num=base64_thread_num,
                input_queue=proc_exec.get_input_queue(),
                output_queue=proc_exec.get_output_queue()))
    elif args.image_type == 'url':
        download_thread_num = args.preprocess_thread_num
        proc_exec.add(
            DownloadProcess(
                thread_num=download_thread_num,
                input_queue=proc_exec.get_input_queue(),
                output_queue=proc_exec.get_output_queue(),
                use_pil_decode=False))

    # transform image data
    proc_exec.add(
        PredictorProcess(
            input_queue=proc_exec.get_input_queue(),
            output_queue=proc_exec.get_output_queue(),
            predict_fn=predict_fn,
            batch_size=batch_size,
            local_rank=args.local_rank,
            thread_num=args.predict_thread_num))

    proc_exec.add(
        DefaultResultFormatProcess(
            input_queue=proc_exec.get_input_queue(),
            output_queue=proc_exec.get_output_queue(),
            reserved_col_names=args.reserved_columns.split(','),
            output_col_names=args.result_column.split(',')))

    # Write result
    output_cols = args.reserved_columns.split(',') + args.result_column.split(
        ',')
    proc_exec.add(
        TableWriteProcess(
            args.output_table,
            output_col_names=output_cols,
            slice_id=worker_id,
            input_queue=proc_exec.get_input_queue()))

    proc_exec.run()
    proc_exec.wait()


def check_args(args, arg_name, default_value=''):
    assert getattr(args, arg_name) != '', f'{arg_name} should not be empty'


def patch_logging():
    # after get_root_logger, logging will not take effect because
    # it sets all other handler to level logging.INFO
    logger = get_root_logger()
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.INFO)


if __name__ == '__main__':
    args = define_args()
    patch_logging()
    if args.launcher:
        init_dist(args.launcher, backend='nccl')
    if args.input_file != '':
        check_args(args, 'output_file')
        build_and_run_file_io(args)
    else:
        check_args(args, 'input_table')
        check_args(args, 'output_table')
        check_args(args, 'image_col')
        check_args(args, 'reserved_columns')
        check_args(args, 'result_column')
        build_and_run_table_io(args)
