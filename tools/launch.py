# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import subprocess
import sys
from argparse import REMAINDER, ArgumentParser


def parse_args():
    """
  Helper function parsing the command line options
  @retval ArgumentParser
  """
    parser = ArgumentParser(description='PyTorch distributed training launch '
                            'helper utilty that will spawn up '
                            'multiple distributed processes')

    # Optional arguments for the launch helper
    parser.add_argument(
        '--nproc_per_node',
        type=int,
        default=1,
        help='The number of processes to launch on each node, '
        'for GPU training, this is recommended to be set '
        'to the number of GPUs in your system so that '
        'each process can be bound to a single GPU.')

    parser.add_argument(
        '--local_mode',
        action='store_true',
        help='If assigned, traning_script should be path of python'
        'script, otherwise python module name')

    # positional
    parser.add_argument(
        'training_script',
        type=str,
        help='The full path to the single GPU training '
        'program/script to be launched in parallel, '
        'followed by all the arguments for the '
        'training script',
        default='tools/train.py')

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    args.node_rank = int(os.environ.get('RANK', '0'))
    args.nnodes = int(os.getenv('WORLD_SIZE', '1'))
    # world size in terms of number of processes
    dist_world_size = args.nproc_per_node * args.nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env['WORLD_SIZE'] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
        current_env['OMP_NUM_THREADS'] = str(1)
        print('*****************************************\n'
              'Setting OMP_NUM_THREADS environment variable for each process '
              'to be {} in default, to avoid your system being overloaded, '
              'please further tune the variable for optimal performance in '
              'your application as needed. \n'
              '*****************************************'.format(
                  current_env['OMP_NUM_THREADS']))

    for local_rank in range(0, args.nproc_per_node):
        # each process's rank
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env['RANK'] = str(dist_rank)
        current_env['LOCAL_RANK'] = str(local_rank)

        # spawn the processes
        cmd = [sys.executable, '-u']
        if not args.local_mode:
            cmd.append('-m')

        cmd.append(args.training_script)

        cmd.append('--local_rank={}'.format(local_rank))

        cmd.extend(args.training_script_args)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=process.returncode, cmd=cmd)


if __name__ == '__main__':
    main()
