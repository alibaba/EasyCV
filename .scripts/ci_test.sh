#!/system/bin/sh

#================================================================
#   Copyright (C) 2022 Alibaba Ltd. All rights reserved.
#
#================================================================

# install requirements
# pip install oss2
# pip install -r requirements.txt

# linter test
pip install -r requirements/tests.txt
# use internal project for pre-commit due to the network problem
if [ `git remote -v | grep alibaba  | wc -l` -gt 1 ]; then
    cp .pre-commit-config.yaml.alibaba  .pre-commit-config.yaml
fi
pre-commit run --all-files
if [ $? -ne 0 ]; then
    echo "linter test failed, please run 'pre-commit run --all-files' to check"
    exit -1
fi

# #setup for git-lfs
# if [ ! -e git-lfs/git_lfs.py ]; then
#  ping gitlab.alibaba-inc.com -c 3
#  # for internal test, use git-lfs
#  if [ $? -eq 0 ]; then
#    git submodule init
#    git submodule update
#  fi
# fi

# #add ossconfig for git-lfs
# OSS_CONFIG=~/.git_oss_config
# if [ ! -e $OSS_CONFIG ]; then
#     echo "$OSS_CONFIG does not exists"
#     exit
# fi

#add ossconfig for unittest
UNITTEST_OSS_CONFIG=~/.ossutilconfig.unittest
if [ ! -e $UNITTEST_OSS_CONFIG ]; then
    echo "$UNITTEST_OSS_CONFIG does not exists"
    exit
fi

export OSS_CONFIG_FILE=$UNITTEST_OSS_CONFIG

# #download test data
# python git-lfs/git_lfs.py pull


export PYTHONPATH=.
export TEST_DIR="/tmp/easycv_test_${USER}_`date +%s`"

# do not uncomments, casue faild in Online UT, install requirements by yourself on UT machine
# pip install -r requirements.txt
#run test
PYTHONPATH=. python tests/run.py
