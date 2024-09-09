cd ./examples/llama3
sh run_pretrain_megatron_llama.sh  \
dsw  \
../../ \
8B   \
2    \
16 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4   \
2  \
full  \
true   \
false  \
false   \
false   \
100  \
/gfshome/llama3-datasets/wudao_llama3bpe_content_document  \
/gfshome/Meta-Llama-3-8B-to-megatron-tp4-pp2  \
409600   \
10000   \
/gfshome/Meta-Llama-outputs
