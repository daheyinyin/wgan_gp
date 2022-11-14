#!/bin/bash

# ============================================================================
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash bash run_train.sh dataroot device_id"
echo "For example: bash run_train.sh /home/cifar10/cifar-10-batches-bin/ 3"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf train
mkdir train
cd ./train
mkdir src
cd ../
cp ./*.py ./train
cp ./src/*.py ./train/src
cd ./train

env > env0.log

echo "train begin."
python train.py --dataroot $1 --device_id $2 > ./train.log 2>&1 &

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
