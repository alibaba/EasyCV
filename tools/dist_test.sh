#!/bin/bash

set -x

CFG=$1
GPUS=$2
CHECKPOINT=$3
PY_ARGS=${@:4}
PORT=${PORT:-29600}

WORK_DIR="$(dirname $CHECKPOINT)/"


# test
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/eval.py \
    $CFG \
    $CHECKPOINT \
     --launcher="pytorch" \
     $PY_ARGS
