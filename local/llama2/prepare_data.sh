cd ./toolkits/model_checkpoints_convertor/llama
sh hf2megatron_convertor.sh \
../../../     \
/gfshome/Llama-2-70b-hf/Llama-2-70b-hf   \
/gfshome/Llama-2-tp8-pp1  \
8  \
1  \
llama2-70b \
0 \
false