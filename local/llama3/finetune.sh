cd ./examples/llama3
sh run_finetune_megatron_llama_withGA.sh  \
dsw  \
../../ \
8B     \
2      \
16     \
1e-5   \
1e-6   \
128   \
128     \
256      \
bf16   \
4      \
2      \
sel    \
true   \
false  \
false  \
false \
300 \
/gfshome/llama3-datasets/alpaca_zh-llama3-train.json   \
/gfshome/llama3-datasets/alpaca_zh-llama3-valid.json   \
/gfshome/Meta-Llama-3-8B-to-megatron-tp4-pp2  \
1000 \
10 \
/gfshome/Meta-Llama-outputs