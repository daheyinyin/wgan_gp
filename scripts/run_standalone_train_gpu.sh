#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train_gpu.sh dataroot device_id "
echo "For example: bash run_standalone_train_gpu.sh /path/to/dataset 0"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

EXEC_PATH=$(pwd)
echo "EXEC_PATH" "$EXEC_PATH"

export DEVICE_ID=$3

rm -rf train
mkdir train
cd ./train
mkdir src
cd ../
cp ../*.py ./train
cp ../src/*.py ./train/src
cd ./train

env > env0.log

echo "train begin."
python train.py --dataroot $1 --device_id $2 --device_target GPU > ./train.log 2>&1 &
