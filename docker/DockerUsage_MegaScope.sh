#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------- Basic distributed parameters ----------
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${1:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"   # Avoid 6002 and can be overridden by environment variables

# ---------- Model Parallelism ----------
TENSOR_MP_SIZE="${TP:-1}"             # TP=1
PIPELINE_MP_SIZE="${PP:-2}"           # PP=2

# ---------- Path ----------
CHECKPOINT_PATH="ngc_models/release_gpt_base"
VOCAB_FILE="datasets/gpt/vocab.json"
MERGE_FILE="datasets/gpt/merges.txt"
DATA_PATH="datasets/gpt/gpt_text_document"

# ---------- Hyperparameters ----------
NUM_LAYERS=16
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=32
SEQ_LEN=2048
MAX_POS=2048
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=32
LR=0.00015
TRAIN_ITERS=20
LR_DECAY_ITERS=320000
LR_DECAY_STYLE=cosine
MIN_LR=1.0e-5
WEIGHT_DECAY=1e-2
LR_WARMUP_FRAC=0.01
CLIP_GRAD=1.0

# ---------- Lightweighted check ----------
[[ -f pretrain_gpt.py ]] || { echo "ERROR: can not find pretrain_gpt.py"; exit 2; }
mkdir -p "$(dirname "$CHECKPOINT_PATH")" || true
for f in "$VOCAB_FILE" "$MERGE_FILE"; do
  [[ -f "$f" ]] || echo "WARN: Lack of file:$f（Possible subsequent error may occur）"
done
[[ -e "$DATA_PATH" ]] || echo "WARN: Data path not exists: $DATA_PATH"

echo "[INFO] GPUS_PER_NODE=$GPUS_PER_NODE NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[INFO] TP=$TENSOR_MP_SIZE PP=$PIPELINE_MP_SIZE"

DISTRIBUTED_ARGS=( --standalone --nproc_per_node "$GPUS_PER_NODE" --max-restarts 0 )

GPT_ARGS=(
  --num-layers "$NUM_LAYERS"
  --hidden-size "$HIDDEN_SIZE"
  --num-attention-heads "$NUM_ATTENTION_HEADS"
  --seq-length "$SEQ_LEN"
  --max-position-embeddings "$MAX_POS"

  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"

  --lr "$LR"
  --train-iters "$TRAIN_ITERS"
  --lr-decay-iters "$LR_DECAY_ITERS"
  --lr-decay-style "$LR_DECAY_STYLE"
  --min-lr "$MIN_LR"
  --weight-decay "$WEIGHT_DECAY"
  --lr-warmup-fraction "$LR_WARMUP_FRAC"
  --clip-grad "$CLIP_GRAD"

  --fp16
  --recompute-method uniform
  --recompute-num-layers 1
  --recompute-granularity full

  --pipeline-model-parallel-size "$PIPELINE_MP_SIZE"
  --tensor-model-parallel-size "$TENSOR_MP_SIZE"
  --transformer-impl local
  --ckpt-format torch
  --no-ckpt-fully-parallel-save
)

DATA_ARGS=(
  --data-path "$DATA_PATH"
  --vocab-file "$VOCAB_FILE"
  --merge-file "$MERGE_FILE"
  --split 949,50,1
)

OUTPUT_ARGS=(
  --log-interval 1
  --save-interval 50
  --eval-interval 20
  --eval-iters 10
)

set -x
torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
  "${GPT_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  --distributed-backend nccl \
  --save "$CHECKPOINT_PATH" \
  --load "$CHECKPOINT_PATH"
set +x
