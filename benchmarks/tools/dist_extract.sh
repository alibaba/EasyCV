#!/usr/bin/env bash
set -x

PYTHON=${PYTHON:-"python"}
CFG=$1
GPUS=$2
WORK_DIR=$3
PY_ARGS=${@:4} # "--checkpoint $CHECKPOINT --pretrained $PRETRAINED"
PORT=${PORT:-29500}

PYTHONPATH=./ $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/extract.py $CFG  --work_dir $WORK_DIR \
    --launcher pytorch ${PY_ARGS}
