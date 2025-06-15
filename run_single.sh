#!/bin/bash

PIPELINE_PARALLEL=4
MODEL_CHUNKS=4

VIRTUAL_STAGE_LAYER=$((16 / (PIPELINE_PARALLEL * MODEL_CHUNKS)))
sed -i "s/^NODE_RANK=[0-9]\+/NODE_RANK=0/" examples/gpt3/train_gpt3_175b_distributed.sh
sed -i "s/^NUM_NODES=[0-9]\+/NUM_NODES=1/" examples/gpt3/train_gpt3_175b_distributed.sh

POD_IP=$(hostname -i)
sed -i "s/^MASTER_ADDR=.*/MASTER_ADDR=$POD_IP/" examples/gpt3/train_gpt3_175b_distributed.sh

sed -i "s/^PIPELINE_PARALLEL=[0-9]\+/PIPELINE_PARALLEL=$PIPELINE_PARALLEL/" examples/gpt3/train_gpt3_175b_distributed.sh
sed -i "s/^VPP=[0-9]\+/VPP=$VIRTUAL_STAGE_LAYER/" examples/gpt3/train_gpt3_175b_distributed.sh

bash examples/gpt3/train_gpt3_175b_distributed.sh
