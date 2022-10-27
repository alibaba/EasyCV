#!/system/bin/sh

#================================================================
#   Copyright (C) 2022 Alibaba Ltd. All rights reserved.
#
#================================================================

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

#add ossconfig for unittest
UNITTEST_OSS_CONFIG=~/.ossutilconfig.unittest
if [ ! -e $UNITTEST_OSS_CONFIG ]; then
    echo "$UNITTEST_OSS_CONFIG does not exists"
    exit
fi

export OSS_CONFIG_FILE=$UNITTEST_OSS_CONFIG
export TEST_DIR="/tmp/easycv_test_${USER}_`date +%s`"

# build package
python setup.py sdist bdist_wheel

# get package path
PACKAGE_PATH=$(ls package/dist/*.whl)

# install easycv
pip uninstall -y pai-easycv
pip install $PACKAGE_PATH

# move source code, ensure import easycv from site-package
mv ./easycv ./easycv_src
#run test
PYTHONPATH=. python tests/run.py
