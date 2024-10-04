cd ~/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
export LD_LIBRARY_PATH=/usr/local/lib64/:$LD_LIBRARY_PATH
sh hf2megatron_convertor.sh \
../../../     \
/gfshome/qwen-ckpts/Qwen1.5-0.5B    \
/gfshome/Qwen1.5-0.5B-hf-to-megatron-tp4-pp2  \
4  \
2  \
qwen1.5-0.5b \
0 \
false