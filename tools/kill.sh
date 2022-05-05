#!/bin/bash
# kill training jobs started by easycv, note it
# will kill all the training processes
kill $(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')
