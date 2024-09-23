#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=4
MASTER_ADDR=10.3.0.25
MASTER_PORT=32131
NNODES=2
NODE_RANK=$RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
TENSOR_MP_SIZE=4
PIPELINE_MP_SIZE=2
VIRTUAL_STAGE_LAYER=1

echo running on rank $NODE_RANK

CHECKPOINT_PATH=/scratch/release_gpt_base
rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH
DATASET_PATH=/root/Pai-Megatron-Patch/local/datasets
VOCAB_FILE=$DATASET_PATH/vocab.json
MERGE_FILE=$DATASET_PATH/merges.txt
DATA_PATH=$DATASET_PATH/gpt-large-cased-vocab-small_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 32 \
    --hidden-size 2560 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 2 \
    --global-batch-size 32 \
    --lr 0.00015 \
    --train-iters 200 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
    --tensor-model-parallel-size $TENSOR_MP_SIZE 
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 200 \
    --eval-iters 10
"

cd /root/Pai-Megatron-Patch/Megatron-LM-231007
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    | tee /gfshome/gpt-outputs/log$RANK.log
