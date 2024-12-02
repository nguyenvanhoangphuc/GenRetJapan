#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Chạy train 
# python run.py --model_name t5-base --code_num 512 --max_length 3 --train_data nq320k/train.json --dev_data nq320k/dev.json --corpus_data nq320k/corpus_lite.json --save_path out/model > genret_train.log
python run.py --model_name sonoisa/t5-base-japanese --code_num 512 --max_length 3 --train_data data_ja/train.json --dev_data data_ja/dev.json --corpus_data data_ja/corpus_lite.json --save_path out_ja_3110/model > genret_train_ja_3110.log
# Evaluate
# python generation.py --model_name sonoisa/t5-base-japanese --code_num 512 --max_length 3 --train_data data_ja/train.json --dev_data data_ja/dev.json --corpus_data data_ja/corpus_lite.json --save_path out_ja/model > genret_train_ja.log