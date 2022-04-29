#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

CFG=$1
GPUS=$2
PY_ARGS=${@:3}
PORT=${PORT:-29527}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

PYTHONPATH=./ $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
tools/train.py $CFG --work_dir $WORK_DIR --launcher pytorch ${PY_ARGS} \
