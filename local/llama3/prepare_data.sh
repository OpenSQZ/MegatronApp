cd ./toolkits/model_checkpoints_convertor/llama
sh hf2megatron_convertor.sh \
../../../     \
/share/zbh/llama3-ckpts/Meta-Llama-3-8B    \
/gfshome/Meta-Llama-3-8B-to-megatron-tp2-pp4  \
2  \
4  \
llama3-8b \
0 \
false
