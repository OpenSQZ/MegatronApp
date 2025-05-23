#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
TENSOR_MP_SIZE=4
PIPELINE_MP_SIZE=2
VIRTUAL_STAGE_LAYER=1

DATA_PATH=datasets/gpt-large-cased-vocab-small_text_document

CHECKPOINT_PATH=ngc_models/release_gpt_base

vocabfile=datasets/vocab.json
mergefile=datasets/merges.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 16 \
       --hidden-size 1536 \
       --num-attention-heads 16 \
       --micro-batch-size 2 \
       --global-batch-size 32 \
       --seq-length 2048 \
       --sequence-parallel \
       --max-position-embeddings 2048 \
       --train-iters 20 \
       --lr-decay-iters 12 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $vocabfile \
       --merge-file $mergefile \
       --data-impl mmap \
       --split 900,50,50 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --recompute-granularity full \
       --recompute-method uniform \
       --log-interval 1 \
       --save-interval 50 \
       --eval-interval 10 \
       --eval-iters 10 \
       --fp16 \
       --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
       --tensor-model-parallel-size $TENSOR_MP_SIZE \
       # --num-layers-per-virtual-pipeline-stage  $VIRTUAL_STAGE_LAYER \
