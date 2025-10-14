export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS=4
export RMOUT="rm -rf /root/mygpt3_random_1_1"
export MEGATRON_CMD="torchrun --nnodes 1 --nproc_per_node $GPUS  --master_addr localhost --master_port 6000 pretrain_gpt.py --num-layers 12 --hidden-size 768 --num-attention-heads 12 --seq-length 2048 --max-position-embeddings 2048 --attention-backend auto --micro-batch-size 1 --global-batch-size 4 --train-iters 40 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.006 --clip-grad 1.0 --fp16 --lr 6.0e-5 --lr-decay-style cosine --min-lr 6.0e-6 --lr-warmup-fraction .001 --lr-decay-iters 430000  --data-path datasets/gpt-large-cased-vocab-small_text_document --vocab-file datasets/vocab.json --merge-file datasets/merges.txt --split 16,5,1 --log-interval 1 --save-interval 10000 --eval-interval 1000 --save /droot/mygpt3_random_1_1 --eval-iters 10 --no-masked-softmax-fusion --load /root/mygpt3_random_"

$RMOUT;${MEGATRON_CMD} --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --transformer-impl local --forward-backward-disaggregating --ignore-forward-tensor-parallel

# --ignore-forward-tensor-parallel