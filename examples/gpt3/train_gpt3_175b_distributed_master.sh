#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=10.233.82.104
MASTER_PORT=6000
NUM_NODES=2
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))
PIPELINE_PARALLEL=2
VPP=2
TENSOR_PARALLEL=4

CHECKPOINT_PATH=ngc_models_gpt                                   #<Specify path>
TENSORBOARD_LOGS_PATH=tensor_board_gpt                           #<Specify path>
VOCAB_FILE=datasets_gpt/vocab.json                               #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=datasets_gpt/merges.txt                               #<Specify path to file>/gpt2-merges.txt
DATA_PATH=datasets/gpt_text_document #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 16
    --hidden-size 2048
    --num-attention-heads 32
    --seq-length 2048
    --max-position-embeddings 2048
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 32
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
    --use-dpp
    --node-ips "192.168.0.7,192.168.0.7,192.168.0.7,192.168.0.7,192.168.0.2,192.168.0.2,192.168.0.2,192.168.0.2"
    --workload $((2048 * 512))
    --num-gpus $GPUS_PER_NODE
    --multi-node
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL
    --pipeline-model-parallel-size $PIPELINE_PARALLEL
    --num-layers-per-virtual-pipeline-stage $VPP
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 50
    --eval-interval 20
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

rm -r $CHECKPOINT_PATH

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
