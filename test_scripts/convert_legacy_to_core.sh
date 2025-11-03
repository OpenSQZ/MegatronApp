
CUDA_DEVICE_MAX_CONNECTIONS=1 \
python tools/checkpoint/convert.py \
  --model-type GPT \
  --loader legacy \
  --saver core \
  --load-dir /path/to/megatron_ckpt/gpt2 \
  --save-dir /path/to/mcore_ckpt/gpt2 \
  --vocab-file /path/to/hf_ckpt/gpt2/vocab.json \
  --megatron-path . \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1