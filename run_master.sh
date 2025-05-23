sed -i "s/^NODE_RANK=[0-9]\+/NODE_RANK=0/" examples/megatron4.0/pretrain_gpt_distributed_small.sh
sed -i "s/^NNODES=[0-9]\+/NNODES=2/" examples/megatron4.0/pretrain_gpt_distributed_small.sh
bash script/benchmark.sh setup v4
bash script/benchmark.sh train
