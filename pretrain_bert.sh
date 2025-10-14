#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=3
# Change for multinode config
MASTER_ADDR=192.168.0.1
MASTER_PORT=6002
NUM_NODES=2
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES+1))
if [ "$NODE_RANK" -eq 0 ]; then
    ((GPUS_PER_NODE++))
fi
echo $GPUS_PER_NODE

CHECKPOINT_PATH=${1:-"checkpoints/bert_fp16"}
TENSORBOARD_LOGS_PATH=${2:-"tensorboard_logs/bert_fp16"}
VOCAB_FILE=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_PATH=${4:-"MOCK"}     # Data prefix, or "MOCK"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

BERT_MODEL_ARGS=(
    --num-layers 24 
    --hidden-size 1024 
    --num-attention-heads 16 
    --seq-length 512 
    --max-position-embeddings 512 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 32 
    --train-iters 1000000 
    --weight-decay 1e-2 
    --clip-grad 1.0 
    --fp16
    --lr 0.0001
    --lr-decay-iters 990000 
    --lr-decay-style linear 
    --min-lr 1.0e-5 
    --weight-decay 1e-2 
    --lr-warmup-fraction .01 
    --clip-grad 1.0 
    --forward-backward-disaggregating
    --ignore-forward-tensor-parallel
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size 2 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py \
    ${BERT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
    