cd ./examples/llama2
sh run_pretrain_megatron_llama.sh  \
dsw  \
../../ \
70B   \
1    \
1 \
1e-5   \
1e-6   \
128  \
128  \
0   \
fp8  \
4   \
2  \
full  \
true   \
false  \
false   \
false   \
300  \
/gfshome/llama2-datasets/wudao_llamabpe_text_document  \
/gfshome/Llama-2-tp4-pp2  \
2048000   \
10000   \
/gfshome/Llama2-outputs