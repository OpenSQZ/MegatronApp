#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_LAUNCH_BLOCKING=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=192.168.0.3
MASTER_PORT=6002
NNODES=2
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
TENSOR_MP_SIZE=2
PIPELINE_MP_SIZE=2
VIRTUAL_STAGE_LAYER=1

CHECKPOINT_PATH=ngc_models/release_gpt_base
VOCAB_FILE=datasets/vocab.json
MERGE_FILE=datasets/merges.txt
DATA_PATH=datasets/gpt-large-cased-vocab-small_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 16 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 2 \
    --global-batch-size 32 \
    --lr 0.00015 \
    --train-iters 20 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1\
    --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
    --tensor-model-parallel-size $TENSOR_MP_SIZE
    --transformer-impl local
"
    # --forward-backward-disaggregating
    # --ignore-forward-tensor-parallel
# granularity
# batch_size

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 50 \
    --eval-interval 20 \
    --eval-iters 10
"

REPORT_NAME="report_baseline$NODE_RANK"

# nsys profile \
#     --stats=true \
#     --trace=cuda,nvtx \
#     -o $REPORT_NAME \
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
