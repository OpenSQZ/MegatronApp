sed -i "s/^NODE_RANK=[0-9]\+/NODE_RANK=0/" examples/megatron4.0/pretrain_gpt_distributed_small.sh
sed -i "s/^NNODES=[0-9]\+/NNODES=1/" examples/megatron4.0/pretrain_gpt_distributed_small.sh
POD_IP=$(hostname -i)
sed -i "s/^MASTER_ADDR=.*/MASTER_ADDR=$POD_IP/" examples/megatron4.0/pretrain_gpt_distributed_small.sh
bash script/benchmark.sh setup v4
bash script/benchmark.sh train
