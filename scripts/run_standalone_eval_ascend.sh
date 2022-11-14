#!/bin/bash

# ============================================================================
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash bash run_eval.sh device_id ckpt_file output_dir"
echo "For example: bash run_standalone_eval_ascend.sh DEVICE_ID CKPT_FILE_PATH OUTPUT_DIR"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
rm -rf eval
mkdir eval
cd ./eval
mkdir src
cd ../
cp ./*.py ./eval
cp ./src/*.py ./eval/src
cd ./eval

env > env0.log

echo "train begin."
python eval.py --device_id $1 --ckpt_file $2 --output_dir $3 > ./eval.log 2>&1 &

if [ $? -eq 0 ];then
    echo "eval success"
else
    echo "eval failed"
    exit 2
fi
echo "finish"
cd ../
