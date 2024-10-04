cd ~/Pai-Megatron-Patch/examples/qwen1_5
rm -rf /scratch/output_megatron_qwen
mkdir -p /gfshome/output_megatron_qwen
mkdir -p /scratch/output_megatron_qwen
sh run_pretrain_megatron_qwen_comm.sh  \
dsw  \
../../ \
0.5B   \
2    \
16 \
1e-5   \
1e-6   \
128  \
128  \
293   \
bf16  \
4   \
2  \
full  \
true   \
false  \
false   \
false   \
100  \
/gfshome/qwen-datasets/wudao_qwenbpe_text_document  \
/gfshome/Qwen1.5-0.5B-hf-to-megatron-tp4-pp2  \
409600   \
10000   \
/scratch/output_megatron_qwen \
| tee /gfshome/output_megatron_qwen/output$RANK.log