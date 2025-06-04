#!/bin/bash

PIPELINE_PARALLEL=4
MODEL_CHUNKS=4

while [[ $# -gt 0 ]]; do
    case $1 in
    --pipeline_parallel)
        PIPELINE_PARALLEL=$2
        shift 2
        ;;
    --model_chunks)
        MODEL_CHUNKS=$2
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

VIRTUAL_STAGE_LAYER=$((32 / (PIPELINE_PARALLEL * MODEL_CHUNKS)))
sed -i "s/^NODE_RANK=[0-9]\+/NODE_RANK=0/" examples/megatron4.0/pretrain_gpt_distributed_small.sh
sed -i "s/^NNODES=[0-9]\+/NNODES=1/" examples/megatron4.0/pretrain_gpt_distributed_small.sh

POD_IP=$(hostname -i)
sed -i "s/^MASTER_ADDR=.*/MASTER_ADDR=$POD_IP/" examples/megatron4.0/pretrain_gpt_distributed_small.sh

sed -i "s/^PIPELINE_MP_SIZE=[0-9]\+/PIPELINE_MP_SIZE=$PIPELINE_PARALLEL/" examples/megatron4.0/pretrain_gpt_distributed_small.sh
sed -i "s/^VIRTUAL_STAGE_LAYER=[0-9]\+/VIRTUAL_STAGE_LAYER=$VIRTUAL_STAGE_LAYER/" examples/megatron4.0/pretrain_gpt_distributed_small.sh

bash script/benchmark.sh setup v4
bash script/benchmark.sh train >trace_pipeline_${PIPELINE_PARALLEL}_model_chunks_${MODEL_CHUNKS}.txt
