#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
# Use PyTorch's recommended variable names to avoid deprecate warnings in your logs
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

# The primary network card in a standalone container is usually eth0. If it is different, please replace it.
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

nvidia-smi -L || true
mkdir -p trace_output /workspace/shared/ckpts_gpt /workspace/shared/tensorboard_gpt

torchrun --standalone --nproc_per_node=4 scripts/training/pretrain_gpt.py \
  --num-layers 16 \
  --hidden-size 2048 \
  --num-attention-heads 32 \
  --seq-length 2048 \
  --max-position-embeddings 2048 \
  --micro-batch-size 2 \
  --global-batch-size 16 \
  --train-iters 5 \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --num-layers-per-virtual-pipeline-stage 2 \
  --untie-embeddings-and-output-weights \
  --no-ckpt-fully-parallel-save \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file datasets/gpt/vocab.json \
  --merge-file datasets/gpt/merges.txt \
  --data-path datasets/gpt_text_document \
  --split 949,50,1 \
  --fp16 \
  --save /workspace/shared/ckpts_gpt \
  --save-interval 50 \
  --tensorboard-dir /workspace/shared/tensorboard_gpt \
  --transformer-impl transformer_engine \
  --lr 3e-4 --min-lr 3e-4 --lr-decay-style constant --lr-warmup-iters 0 \
  --trace --trace-dir trace_output --trace-interval 5 \
  --continuous-trace-iterations 2 --trace-granularity full
