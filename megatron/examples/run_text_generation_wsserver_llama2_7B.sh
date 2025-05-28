#!/bin/bash
# This example will start serving the Llama2-7B model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_wsserver.py   \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model llama2-ckpts/Llama-2-7b-chat-megatron/tokenizer.model \
       --load llama2-ckpts/Llama-2-7b-chat-megatron \
       --exit-on-missing-checkpoint \
       --use-checkpoint-args \
       --no-load-optim \
       --no-load-rng \
       --fp16 \
       --untie-embeddings-and-output-weights \
       --use-rotary-position-embeddings \
       --normalization RMSNorm \
       --no-position-embedding \
       --no-masked-softmax-fusion \
       --micro-batch-size 1
