#!/bin/bash

# Runs the "340M" parameter model (Bert - Large)

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=10.233.82.104
MASTER_PORT=6000
NUM_NODES=2
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE * $NUM_NODES))
PIPELINE_PARALLEL=2
VPP=2
TENSOR_PARALLEL=4

CHECKPOINT_PATH=ngc_models_bert #<Specify path>
TENSORBOARD_LOGS_PATH=tensor_board_bert  #<Specify path>
VOCAB_FILE=datasets_bert/vocab.txt #<Specify path to file>/bert-vocab.json
DATA_PATH=datasets/bert_text_sentence #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
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
    --micro-batch-size 2 
    --global-batch-size 32 
    --train-iters 20 
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

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py \
    ${BERT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
    