sed -i "s/^NODE_RANK=[0-9]\+/NODE_RANK=1/" examples/gpt3/train_gpt3_175b_distributed_worker.sh
sed -i "s/^NUM_NODES=[0-9]\+/NUM_NODES=2/" examples/gpt3/train_gpt3_175b_distributed_worker.sh
bash examples/gpt3/train_gpt3_175b_distributed_worker.sh
