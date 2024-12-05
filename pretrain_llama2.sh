cd examples/llama2/
sh run_pretrain_megatron_llama.sh  \
dsw  \
../../ \
7B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
0   \
bf16  \
1   \
4  \
sel  \
true   \
false  \
false   \
false   \
100000  \
/gfshome/llama2-datasets/wudao_llamabpe_text_document   \
/gfshome/llama2-ckpts/Llama-2-7b-hf-to-megatron-tp1-pp4   \
40960   \
10000   \
/mnt/output_megatron_llama2   \
| tee /root/exp/Pai-Megatron-Patch/results/output$RANK.log