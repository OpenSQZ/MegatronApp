sed -i "s/^NODE_RANK=[0-9]\+/NODE_RANK=0/" examples/gpt3/train_gpt3_175b_distributed_master.sh
sed -i "s/^NUM_NODES=[0-9]\+/NUM_NODES=2/" examples/gpt3/train_gpt3_175b_distributed_master.sh
POD_IP=$(hostname -i)
sed -i "s/^MASTER_ADDR=.*/MASTER_ADDR=$POD_IP/" examples/gpt3/train_gpt3_175b_distributed_master.sh
sed -i "s/^MASTER_ADDR=.*/MASTER_ADDR=$POD_IP/" examples/gpt3/train_gpt3_175b_distributed_worker.sh
bash examples/gpt3/train_gpt3_175b_distributed_master.sh
