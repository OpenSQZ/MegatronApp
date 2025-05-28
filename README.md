Before running, make sure that websockets is installed.
```
pip install websockets
```
Moreover, save the megatron-style model file to the desired directory (`megatron/llama2-ckpts/Llama-2-7b-chat-megatron/` or `megatron/llama2-ckpts/Llama-2-13b-chat-megatron/`).

To run the backend model inference, use
```
cd megatron
bash examples/run_text_generation_wsserver_llama2_7B.sh
bash examples/run_text_generation_wsserver_llama2_13B.sh
```

To run the frontend web page, use
```
cd transformer-visualize
nvm install --lts
npm run dev
```
In the web page, you can provide a batch of prompts to generate, then it will return the intermediate results and plots.