# MegatronApp Quick Start

This guide explains how to prepare environments, convert checkpoints, and test different components of **MegatronApp** including **MegaScope**, **MegaScan**, **MegaDPP**, and **MegaFBD**.

---

## ⚠️ Important Notes

* **NVIDIA Container Toolkit** must be installed and configured if you are running inside Docker with GPUs.
* A **large shared memory (shm)** mount is required for MegaDPP.
* The provided Docker image includes **SSH login**.
  For security, either:

  * Disable the SSH service, or
  * Configure it to accept **key-based authentication only**.

---

## 1. Test MegaScope

### Prepare a HuggingFace checkpoint

```bash
huggingface-cli download openai-community/gpt2-medium \
  --local-dir /path/to/gpt2-medium \
  --local-dir-use-symlinks False
```

### Convert checkpoints

* `test_scripts/gpt2_convert.py` has been modified to require `iteration > 0`.
* `tools/checkpoint/loader_legacy.py` has been patched to set/unset global `margs` before and after `load_args_from_checkpoint`.

Run the following after adjusting paths in scripts:

```bash
bash test_scripts/convert_hf_to_megatron.sh
bash test_scripts/convert_legacy_to_core.sh
```

### Run text generation server

Update paths in `test_text_generation_server_gpt2.sh`, then:

```bash
bash test_scripts/test_text_generation_server_gpt2.sh
```

**Tokenizer fix:**

* Newer Megatron’s `_GPT2BPETokenizer` does not provide `decoder` like HuggingFace.
* Added `decoder`, `encoder`, `offsets` methods, and `_LazyReadableDecoder` in `megatron/training/tokenizer/tokenizer.py`.

### Launch MegaScope frontend

```bash
cd transformer-visualize

# Optional: if your Node.js environment changed
# rm -rf ./node_modules
# npm install

npm run dev
```

Access [http://localhost:5173/](http://localhost:5173/).

* If running on a remote server, use **port forwarding** to your local machine.
* If VM ports are not exposed, configure backend port mapping or update the IP/port in Vue.js frontend WebSocket config.

---

## 2. Test MegaScan

### Prepare dataset

```bash
bash test_scripts/process_dataset_gpt.sh
```

### Run small training with tracing

Modify path parameters in `train_gpt_single_trace.sh`, then:

```bash
bash test_scripts/train_gpt_single_trace.sh
```

Trace files are saved at:

```
MegatronApp/trace_output/benchmark-data-0-pipeline-0-tensor-0.json
```

### Aggregate trace

```bash
python scripts/aggregate.py --b trace_output --output test_scripts/trace_output.json
# or
bash test_scripts/aggregate_trace.sh
```

Upload results to [Perfetto UI](https://ui.perfetto.dev/) for analysis.

---

## 3. Test MegaDPP (Single Node, 4 GPUs)

Requirements:

* Pod with **4 GPUs**
* Large **/dev/shm** allocation

### Run training

* Set `MASTER_ADDR` to the **node’s IP address**.
* **Do not modify `--node-ips`.**
* Ensure `--use-dpp` and other DPP parameters remain.

Adjust **TP, PP, VPP, MBS, GBS** inside `test_train_gpt_single_dpp.sh`, then run:

```bash
bash test_scripts/test_train_gpt_single_dpp.sh
```

---

## 4. Test MegaFBD (Multi Node, 2×4 GPUs)

Requirements:

* Two pods, each with **4 GPUs**

### Run training

On node rank 0:

```bash
bash test_scripts/test_train_gpt_distributed_fbd.sh 0
```

On node rank 1:

```bash
bash test_scripts/test_train_gpt_distributed_fbd.sh 1
```

Make sure `MASTER_ADDR` is set to the actual IP of one node.

---

✅ You are now ready to explore **MegaScope visualization**, **MegaScan tracing**, **MegaDPP intra-node parallelism**, and **MegaFBD multi-node distributed training**.

