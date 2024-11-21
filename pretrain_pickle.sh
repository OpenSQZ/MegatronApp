export OUTPUT_ARGS=" " # --only-serialization"
export PICKLE="5"
SAVE_INTERVAL=200
cd ./examples/llama3
rm -rf /scratch/Meta-Llama-outputs
mkdir -p /gfshome/Meta-Llama-outputs
mkdir -p /scratch/Meta-Llama-outputs
sh run_pretrain_megatron_llama_pickle.sh  \
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
2  \
full  \
true   \
false  \
false   \
false   \
${SAVE_INTERVAL}  \
/gfshome/llama3-datasets/wudao_llama3bpe_content_document  \
/gfshome/Meta-Llama-3-8B-to-megatron-tp2-pp2  \
409600   \
10000   \
/scratch/Meta-Llama-outputs \