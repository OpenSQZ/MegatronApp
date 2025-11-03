
python test_scripts/gpt2_convert.py \
  --load_path /path/to/hf_ckpt/gpt2 \
  --save_path /path/to/megatron_ckpt/gpt2 \
  --megatron-path . \
  --print-checkpoint-structure \
  --target_tensor_model_parallel_size 1 \
  --target_pipeline_model_parallel_size 1 \
  --target_data_parallel_size 1 \
  --target_params_dtype fp32 \
  --make_vocab_size_divisible_by 1