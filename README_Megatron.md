<div align="center">

**MegatronDPP (Megatron with Dynamic Pipeline Parallel)**

**📢 Annoncements**

> Megatron DPP has integrated multi-node training with tensor parallel support

**🌍 Choose Your Language**

[English](README_Megatron.md)

</div>

# 🌟 Overview

An integration of a dynamic pipeline-parallel algorithm to the Megatron distributed training framework, along with fully parallelizable P2P communication between different pipeline ranks via shared memory (same node) or RDMA (different nodes)

# ✨ Core Features

- Supports a dynamic pipeline-parallel algorithm that selects the next microbatch to compute via a greedy rule.

- Supports tensor transfer between GPUs locally via shared memory.

- Supposts tensor transfer between GPUs on different nodes via remote direct memory access (RDMA).

- Supports a flag `--use-dpp` to switch between dynamic algorithm and original pipeline algorithm.

- Supports flags `--multi-node` and `--node-ips` to pass in the infiniband IPs of different nodes for multi-node training.

- Supports various levels of data-parallel, pipeline-parallel and tensor-parallel.

# 📥 Supported Data Sources & Language Models

We provide demo examples for the following models. See the files `run_{single,master,worker}_<model>.sh`
| Data Sources You Can Add | Supported Language Models |
| ------------------------ | ------------------------- |
| Sample dataset provided & self-chosen                      | GPT                       |
| Sample dataset provided & self-chosen                      | BERT                       |

To run other models, please refer to the `examples/` directory to adjust the configurations.

# 🚀 Quickstart

## Environment Configuration

- The following is the pod configuration.

```yaml
ContainerImage: ngc.nju.edu.cn/nvidia/pytorch:25.03-py3
GPU: RTX4090

NVMEStorage: 50G 
Limits:
  CPU: 28
  memory: 100Gi
  GPU: 4
UseShm: true
ShmSize: 16Gi

UseIB: true
```

- The python environment in the image automatically includes almost all of the required packages, to install additional required packages, run

```bash
pip install -r requirements.txt
```

- Install infiniband prerequisites

```bash
bash prerequisite.sh
```

- Build the `shm_tensor_new_rdma` (for multinode) and `shm_tensor_new_rdma_pre_alloc` module.

```bash
cd megatron/shm_tensor_new_rdma
pip install -e .
```

```bash
cd megatron/shm_tensor_new_rdma_pre_alloc
pip install -e .
```

## Run

### Dataset Preparation

The dataset preparation step follows largely from the Megatron framework.

First, prepare your dataset in the following `.json` format with one sample per line

```json
{"src": "bloomberg", "text": "BRIEF-Coach Inc launches tender offer to acquire Kate Spade & Co for $18.50 per share in cash. May 26 (Reuters) - Coach Inc: * Coach Inc launches tender offer to acquire Kate Spade & Company for $18.50 per share in cash * Coach Inc launches tender offer to acquire kate spade & company for $18.50 per share in cash * Coach Inc - tender offer will expire at 11:59 P.M. Edt on June 23, 2017, unless extended * Coach Inc - Chelsea Merger Sub Inc, has commenced a tender offer for all of outstanding shares of common stock, par value $1.00 per share, of Kate Spade & Company Source text for Eikon: Further company coverage: May 26 (Reuters) - Coach Inc: * Coach Inc launches tender offer to acquire Kate Spade & Company for $18.50 per share in cash * Coach Inc launches tender offer to acquire kate spade & company for $18.50 per share in cash * Coach Inc - tender offer will expire at 11:59 P.M. Edt on June 23, 2017, unless extended * Coach Inc - Chelsea Merger Sub Inc, has commenced a tender offer for all of outstanding shares of common stock, par value $1.00 per share, of Kate Spade & Company Source text for Eikon: Further company coverage:", "type": "Eng", "id": "0", "title": "BRIEF-Coach Inc launches tender offer to acquire Kate Spade & Co for $18.50 per share in cash. "}
{"src": "bloomberg", "text": "Var Energi agrees to buy Exxonmobil's Norway assets for $4.5 bln. MILAN, Sept 26 (Reuters) - Var Energi AS, the Norwegian oil and gas group 69.6% owned by Italian major Eni, has agreed to buy the Norwegian upstream assets of ExxonMobil for $4.5 billion. The deal is expected to be completed in the final quarter of this year, Var Energi said on Thursday. Reporting by Stephen Jewkes; editing by Francesca Landini MILAN, Sept 26 (Reuters) - Var Energi AS, the Norwegian oil and gas group 69.6% owned by Italian major Eni, has agreed to buy the Norwegian upstream assets of ExxonMobil for $4.5 billion. The deal is expected to be completed in the final quarter of this year, Var Energi said on Thursday. Reporting by Stephen Jewkes; editing by Francesca Landini", "type": "Eng", "id": "1", "title": "Var Energi agrees to buy Exxonmobil's Norway assets for $4.5 bln. "}
{"src": "bloomberg", "text": "Trump says 'incorrect' he is willing to meet Iran with 'no conditions'. WASHINGTON (Reuters) - U.S. President Donald Trump on Sunday appeared to play down the chances that he might be willing to meet with Iranian officials, saying reports that he would do so without conditions were not accurate. \u201cThe Fake News is saying that I am willing to meet with Iran, \u2018No Conditions.\u2019 That is an incorrect statement (as usual!),\u201d Trump said on Twitter. In fact, as recently as on Sept. 10, U.S. Secretary of State Mike Pompeo said \u201cHe (Trump) is prepared to meet with no preconditions.\u201d Reporting By Arshad Mohammed; Editing by Shri Navaratnam WASHINGTON (Reuters) - U.S. President Donald Trump on Sunday appeared to play down the chances that he might be willing to meet with Iranian officials, saying reports that he would do so without conditions were not accurate. \u201cThe Fake News is saying that I am willing to meet with Iran, \u2018No Conditions.\u2019 That is an incorrect statement (as usual!),\u201d Trump said on Twitter. In fact, as recently as on Sept. 10, U.S. Secretary of State Mike Pompeo said \u201cHe (Trump) is prepared to meet with no preconditions.\u201d Reporting By Arshad Mohammed; Editing by Shri Navaratnam", "type": "Eng", "id": "2", "title": "Trump says 'incorrect' he is willing to meet Iran with 'no conditions'. "}
```
note that we have provided a sample dataset under `datasets_gpt/` and `datasets_bert/`.

Then, prepare the vocab file (gpt and bert) and the merges file (gpt-only). We have provided it in the respective directories.

For bert, run the following
```bash
cd datasets
python ../tools/preprocess_data.py \
       --input ../datasets_bert/dataset.json \
       --output-prefix bert \
       --vocab-file ../datasets_bert/vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
       --workers $(nproc)
```
where the paths can be changed according to the location of your files and the place where you want the generated files to be.

For GPT, run the following
```bash
cd datasets
python ../tools/preprocess_data.py \
       --input ../datasets_gpt/dataset.json \
       --output-prefix gpt \
       --vocab-file ../datasets_gpt/vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ../datasets_gpt/merges.txt \
       --append-eod
       --workers $(nproc)
```

For other models, please refer to `nvidia/megatron` for the corresponding datasets.

### Single Node Distributed Training
To run distributed training on a single node, go to the project root directory and run

```bash
bash run_single_gpt.sh
```

for GPT and

```bash
bash run_single_bert.sh
```

for bert.

The `run_single_<model>.sh` files have the following structure:

- Parameters include `pipeline_parallel`, `model_chunks` and `tensor_parallel`
- The `virtual_stage_layer` parameter sets how many layers are there in a single virtual pipeline stage. It is calculated as
$$
\frac{\text{total layer of model}}{\text{pipeline parallel}\times\text{model chunks}}
$$
where total layer is set under `examples/` under the corresponding model.
- It gets the IP address of the pod and writes it to the shell script.
- Finally it runs the shell script under the corresponding model under `examples/`

There are also several critical parameters in `examples/gpt3/train_gpt3_175b_distributed.sh` (bert model under the corresponding `bert/` directory)

- `--use-dpp` switches to DPP algorithm
- `--workload` specifies the workload of each single thread, and hence determines the number of threads used in P2P communication
- `--num-gpus` specify the number of GPUs on the current node (single node training)
- Other critical parameters include the number of layers of the model (note that currently the value is 16 and is static in `run_single_<model>.sh`, needs to simultaneously modify `run_single_<model>.sh` if adjusting the layers), the global batch size and the sequence length

For the remaining models, you can either directly run
```bash
bash examples/<model>/<train_file>.sh
```
or write a file similar to `run_{single,master,worker}_<model>.sh` that sets up configurations and runs the shell under `examples/`

### Multinode Distributed Training
To run distributed training on multiple nodes, go to the root directory. First run

```bash
bash run_master_<model>.sh
```

and then start another pod and run

```bash
bash run_worker_<model>.sh
```

The `run_master_<model>.sh` has the following parameters

- Similar to `run_single_<model>.sh`, we have `pipeline_parallel`, `model_chunks` and `tensor_parallel`
- It writes the master pod IP to `examples/gpt3/train_gpt3_175b_distributed_master.sh` and to `train_gpt3_175b_distributed_worker.sh` (bert in the corresponding directory)
- Set the number of nodes to be 2 and master node has rank 0
- Starts the shell under `examples`

and `run_worker_<model>.sh` does the following
- Set the number of nodes to be 2 and the worker node has rank 1
- Starts the shell under `examples`

The `examples/gpt3/train_gpt3_175b_distributed_master.sh` and `examples/gpt3/train_gpt3_175b_distributed_worker.sh` is similar to the single node version, except that the `--node-ips` is mandatory, which is the infiniband IPs of the pods in the order of their GPU ranks. And also the `--multi-node` flag should be turned on.

### Profiling

Each run will generate a trace dir in `benchmark`. Go to the `profiling` directory and run

```python
python aggregate.py --benchmark_dir benchmark/your-benchmark-dir
```

in the root dir to produce an aggregated trace file.
