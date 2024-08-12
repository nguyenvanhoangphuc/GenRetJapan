#!/bin/bash

# Set CUDA device to 1
export CUDA_VISIBLE_DEVICES=0

# Disable NCCL P2P and IB for compatibility with RTX 4000 series
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Chạy train 
python run.py --model_name t5-base --code_num 512 --max_length 3 --train_data nq320k_bm25_split/train.json --dev_data nq320k_bm25_split/dev.json --corpus_data nq320k_bm25_split/corpus_lite.json --save_path out_bm25/model
