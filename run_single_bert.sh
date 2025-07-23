#!/bin/bash

PIPELINE_PARALLEL=4
MODEL_CHUNKS=4
TENSOR_PARALLEL=1
VIRTUAL_STAGE_LAYER=$((16 / (PIPELINE_PARALLEL * MODEL_CHUNKS)))
sed -i "s/^PIPELINE_PARALLEL=[0-9]\+/PIPELINE_PARALLEL=$PIPELINE_PARALLEL/" examples/bert/train_bert_340m_distributed.sh
sed -i "s/^VPP=[0-9]\+/VPP=$VIRTUAL_STAGE_LAYER/" examples/bert/train_bert_340m_distributed.sh
sed -i "s/^TENSOR_PARALLEL=[0-9]\+/TENSOR_PARALLEL=$TENSOR_PARALLEL/" examples/bert/train_bert_340m_distributed.sh

POD_IP=$(hostname -i)
sed -i "s/^MASTER_ADDR=.*/MASTER_ADDR=$POD_IP/" examples/bert/train_bert_340m_distributed.sh
sed -i "s/^NODE_RANK=[0-9]\+/NODE_RANK=0/" examples/bert/train_bert_340m_distributed.sh
sed -i "s/^NUM_NODES=[0-9]\+/NUM_NODES=1/" examples/bert/train_bert_340m_distributed.sh

bash examples/bert/train_bert_340m_distributed.sh
