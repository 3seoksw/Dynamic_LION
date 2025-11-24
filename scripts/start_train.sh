#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=true
GPUS=${GPUS:-0}
NPROC=${NPROC:-1}
PORT=${PORT:-12345}

CFG=${CFG:-configs/lion_train_stage4.yaml}

CUDA_VISIBLE_DEVICES=${GPUS} torchrun --master_port ${PORT} --nproc_per_node=${NPROC} train.py --cfg-path ${CFG}

