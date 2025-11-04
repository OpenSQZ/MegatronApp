output_dir="/path/to/datasets/gpt/"
mkdir -p $output_dir

python tools/preprocess_data.py \
  --input datasets/gpt/dataset.json \
  --output-prefix $output_dir/bloomberg \
  --vocab-file datasets/gpt/vocab.json \
  --tokenizer-type GPT2BPETokenizer \
  --merge-file datasets/gpt/merges.txt \
  --append-eod \
  --json-keys text \
  --workers $(nproc)
