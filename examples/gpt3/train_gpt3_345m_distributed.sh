#!/bin/bash

# Runs the "345m" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=checkpoints/gpt3_345m_distributed
TENSORBOARD_LOGS_PATH=checkpoints/tb_logs/gpt3_345m_distributed
VOCAB_FILE=datasets/gpt/vocab.json
MERGE_FILE=datasets/gpt/merges.txt
DATA_PATH=datasets/gpt/gpt_text_document # modify this to your dataset path

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12 
    # --hidden-size 512 
    --hidden-size 128
    --num-attention-heads 8 
    --seq-length 1024 
    --max-position-embeddings 2048 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 64
    --global-batch-size 512
    # --rampup-batch-size 16 16 5859375 
    --train-iters 20
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --trace
    --trace-dir trace_output
    --trace-interval 5
    --continuous-trace-iterations 2
    --trace-granularity full
    --transformer-impl local
    # --sequence-parallel
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2 
	--pipeline-model-parallel-size 2 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} scripts/training/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
