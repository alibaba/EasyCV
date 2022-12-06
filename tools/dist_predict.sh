#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}

GPUS=$1
PY_ARGS=${@:2}
PORT=${PORT:-29527}

PYTHONPATH=./ $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
tools/predict.py $PY_ARGS --launcher pytorch \
