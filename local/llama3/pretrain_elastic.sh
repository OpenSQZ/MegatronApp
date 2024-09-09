cd ./examples/llama3
sh run_pretrain_megatron_llama_elastic.sh  \
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
2   \
4  \
full  \
true   \
false  \
false   \
false   \
100  \
/gfshome/llama3-datasets/wudao_llama3bpe_content_document  \
/gfshome/Meta-Llama-3-8B-to-megatron-tp2-pp4  \
409600   \
10000   \
/gfshome/Meta-Llama-outputs
