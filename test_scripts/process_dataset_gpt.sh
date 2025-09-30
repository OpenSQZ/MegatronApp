output_dir="/path/to/datasets_gpt/"
mkdir -p $output_dir

python tools/preprocess_data.py \
  --input datasets_gpt/dataset.json \
  --output-prefix $output_dir/bloomberg \
  --vocab-file datasets_gpt/vocab.json \
  --tokenizer-type GPT2BPETokenizer \
  --merge-file datasets_gpt/merges.txt \
  --append-eod \
  --json-keys text \
  --workers $(nproc)
