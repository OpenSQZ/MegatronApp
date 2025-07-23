sed -i "s/^NODE_RANK=[0-9]\+/NODE_RANK=1/" examples/bert/train_bert_340m_distributed_worker.sh
sed -i "s/^NUM_NODES=[0-9]\+/NUM_NODES=2/" examples/bert/train_bert_340m_distributed_worker.sh
bash examples/bert/train_bert_340m_distributed_worker.sh
