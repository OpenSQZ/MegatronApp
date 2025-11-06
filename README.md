<div align="center">
  <img src="images/megatronapp.png" alt="MegatronApp logo" height="96">
</div>
<h1 align="center">MegatronApp: Toolchain Built around Megatron-LM for Distributed Training</h1>

<p align="center">
  An extension for performance tuning, slow-node detection, and training-process visualization.
</p>

<p align="center">
  <a href="https://github.com/OpenSQZ/MegatronApp/blob/main/docker/DockerUsage.md">🍳 Cookbook</a> |
  <a href="https://arxiv.org/pdf/2507.19845">📄 Technical Report</a>
</p>

</div>

<!-- **📢 Announcements**  

> List 2~3 important annoncements, including milestones, major releases... -->

<!-- **🌍 Choose Your Language**

[English](README.md)  | [中文](cn/README.md)  -->

</div>

# News <!-- omit in toc -->

### 📌 Pinned
* [2025.10.17] 🔥🔥🔥 We provide user-friendly [docker guidance](./docker/DockerUsage.md) for all four features of MegatronApp. Please try it out!
* [2025.07.27] 📢📢📢 The MegatronApp technical report has been released! See [here](https://arxiv.org/pdf/2507.19845).
* [2025.07.04] 🔥🔥🔥 MegatronApp is officially launched at WAIC 2025! Our code is available [here](https://github.com/OpenSQZ/MegatronApp). Come and try it out!


# 🔥 Demo
**MegaScan**

<img width="540" height="295" alt="image" src="https://github.com/user-attachments/assets/393e833d-d356-4611-82bb-ab782874e92c" />

<img width="540" height="295" alt="image" src="https://github.com/user-attachments/assets/d00c387c-78eb-42eb-8f13-8001782c9e9e" />


**MegaScope**

<img width="540" height="295" alt="image" src="https://github.com/user-attachments/assets/ac39289d-de2c-474c-a173-b4bd23e98299" />



# 🌟 Overview
<!-- TraceMegatron is a tool designed to identify and diagnose performance bottlenecks in distributed training environments, particularly in the context of Megatron-LM. It helps visualize the training process in a global view and gives a heuristic-based algorithm to pinpoint the slowest rank with the most probable cause of the slowdown. This tool is particularly useful for developers and researchers working with large-scale distributed training systems, enabling them to optimize their training processes and improve overall efficiency. -->

MegatronApp is a toolchain built around the Megatron-LM training framework, designed to give practitioners a suite of value-added capabilities such as performance tuning, slow-node detection, and training-process visualization.

The project currently offers four core modules:

- MegaScan is a low-overhead tracing and anomaly detection system designed on Megatron-LM for large-scale distributed training. Detecting and locating hardware performance anomalies, such as GPU downclocking, is extremely challenging in large distributed environments. A single slow GPU can cause a cascading delay, degrading the performance of the entire cluster and making it difficult to pinpoint the source. This module aims to solve this problem by capturing and analyzing runtime trace data. By providing a global, high-precision view of all operations across all GPUs, MegaScan can identify specific patterns caused by hardware anomalies, allowing for accurate detection and root cause localization.
- MegaFBD (Forward-Backward Decoupling) – Automatically splits the forward and backward phases onto different devices to resolve imbalances in compute, communication, and memory usage between the two stages, optimizing resource allocation and boosting overall utilization.
- MegaDPP (Dynamic Pipeline Planning) – Dynamically optimizes pipeline-parallel scheduling during training, allowing each device to adjust its schedule in real time according to progress, deferring selected compute or transfer steps to alleviate network pressure.
- MegaScope – Dynamically captures, processes, and caches intermediate results during training according to user-defined metrics, then displays them through an interactive visualization interface. MegaScope aims to make the "black box" of Large Language Models transparent. With this tool, users can observe and analyze things that happen inside a model as it processes text, such as how attention scores and output probabilities are distributed, how the vector representations change among different tokens and prompts.

The four modules are fully isolated and integrated into the Megatron-LM codebase as plugins; users can flexibly enable or disable any of them at launch via control flags.

The technical report of MegatronApp can be seen [here](./MegatronApp.pdf).

# ✨ Core Features
<!-- List 3~5 core features of the project.

> Sample    
> -  📊 Centralized Health Data Input: Easily consolidate all your health data in one place.    
> -  🛠️ Smart Parsing: Automatically parses your health data and generates structured data files.     
> -  🤝 Contextual Conversations: Use the structured data as context for personalized interactions with GPT-powered AI. -->

<!-- - 🔍 **Lightweight Tracing**: Provides detailed tracing of the training process, capturing performance metrics and identifying potential bottlenecks in distributed training environments. 

- 📊 **Global Visualization**: Provides a comprehensive overview of the training process across all ranks, allowing users to visualize the Computation and Communication of distributing training and identify potential bottlenecks.

- 🏎️ **Heuristic-Based Diagnosis**: Implements a heuristic-based algorithm to pinpoint the slowest rank and the most probable cause of the slowdown, enabling fast handling of performance issues. -->

## MegaScan

📊 Low-Overhead Tracing: Utilizes CUDA Events for high-precision, asynchronous timing of operations with minimal impact on training performance (approx. 10% overhead in tests).

🛠️ Automated Data Pipeline: Automatically aggregates trace files from all distributed ranks, reconstructs communication dependencies, and aligns scattered timelines into a single, globally consistent view.

🧠 Heuristic Detection Algorithm: Implements a multi-stage heuristic algorithm to detect and locate faults like GPU downclocking by comparing peer operations across parallel dimensions and analyzing communication behavior.

🖥️ Rich Visualization: Generates trace files in the Chrome Tracing Format, allowing for intuitive, interactive visualization and analysis of complex distributed training runs using standard tools like chrome://tracing and Perfetto UI.

## MegaScope

📊 Real-time generation and visualization: Input any prompt and watch the model generate text token by token, with its internal states displayed in sync.

🛠️ Intermediate result visualization: 

-  Display key intermediate variables like QKV vectors and MLP layer outputs as heatmaps.
-  Attention matrix analysis: Freely select any layer and attention head to view its dynamic attention weight distribution.
-  Output probability visualization: At each generate-next-token step, show the sampled token and its probability, along with other top-k candidates, revealing the model's decisions.

🧠 Interactive analysis: 

-  A rich set of interactive controls allows users to easily switch between different visualization dimensions.
-  PCA dimensionality reduction: Project high-dimensional vector representations onto a 2D space to analyze the similarities and differences between tokens and prompts.

🖥️ Model perturbation injection: To facilitate in-depth research on model robustness, we provide several model perturbation features.
- Storage perturbation: Inject noise into critical model parameters to simulate the error in storage devices.
- Calculation perturbation: Inject noise during the model's forward pass (e.g. at the output of MLP layer).
- System perturbation: Simulate a constant error between each transformer layer.
Through the UI, users can precisely control the location, activation, type and extent of the perturbations.

### MegaDPP

📊 A dynamic pipeline-parallel scheduling algorithm: It selects the next microbatch to compute via a customized greedy rule based on user requirements:

- Depth-first computation: give priority to computing the same data on different model chunks for lower GPU memory usage
- Breadth-first computation: give priority to computing different data on the same model chunks for lower communication contention

🛠️ An efficient shared-memory based communication library:

- Concurrent asynchronous send/recv operations
- Dynamically track the completion status of operations

For more details, see [README_Megatron.md](README_Megatron.md)

### MegaFBD
📊 Instance-Level Decoupled Scheduling: The forward and backward phases are split into two logical processes, each assigned a different rank and bound to separate resources to reduce coupling.

🛠️ Heterogeneous Resource Mapping Optimization: The forward phase can be deployed on lightly loaded devices or CPUs, alleviating GPU pressure.

🧠 Differentiated Parallelism Configuration: Considering factors like activation reuse and communication volume, the forward phase is assigned a lower degree of parallelism to reduce communication overhead.

🖥️ Thread-Level Coordination Mechanism: A communication coordinator ensures necessary data synchronization between forward and backward phases, avoiding deadlocks and redundant communication.



# 🗺️ Tech Architecture / Project Diagram / Workflow
<!-- Illustrate the key technical points with technical architecture, workflow and so on. -->

MegatronApp uses a decoupled frontend-backend architecture with WebSockets to enable low-latency, real-time data communication between the model backend and the visualization frontend.

- Frontend: Based on Vite+Vue+TypeScript, rendering all interactive charts and controls.
- Backend: Based on Megatron, responsible for hosting the LLM. It uses flags to control the extraction of intermediate results during a forward pass, which maintains low time overhead when visualization function is not enabled.
- Communication: The frontend and backend are connected via a WebSocket.


<!-- # 📥 Supported Data Sources & Language Models

| Data Sources You Can Add	| Supported Language Models |
|------------------|--------------------
|xxx	|xxx       |
|xxx	|xxx       | -->

# 🚀 Quickstart
<!-- Include the following information, and provide necessary samples and datasets. You can adjust content based on your project accordingly.

- Install and deploy (lists prerequisites if any)
- Set up and configure (provides examples and datasets if any)
- Build and run
- Access with browser(optional)
- Verify the result -->

## Docker (Recommended)

We strongly recommend using the release of [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for installation. This container comes with all dependencies pre-installed with compatible versions and optimized configurations for NVIDIA GPUs.

```bash
# Run container with mounted directories
docker run --runtime --nvidia --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -v /path/to/dataset:/workspace/dataset \
  -v /path/to/checkpoints:/workspace/checkpoints \
  nvcr.io/nvidia/pytorch:25.04-py3
```



To install additional required packages, run

```bash
pip install -r requirements/requirements.txt
```

## MegaScan

We provide a basic repro for you to quickly get started with MegaScan.

0. Data preparation:

Please refer to [README_Megatron.md](README_Megatron.md) section "Dataset Preparation" and Nvidia's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for more details.

1. Run NVIDIA’s Megatron-LM training with MegaScan enabled by adding the following command line arguments:

```bash
--trace
--trace-dir trace_output
--trace-interval 5 # optional, default is 5 iterations
--continuous-trace-iterations 2 # optional, default is 2 iterations
--trace-granularity full # optional, default is full
--transformer-impl local # currently only supports local transformer implementation
```

`examples/gpt3/train_gpt3_345m_distributed.sh` is an example script. You can modify the script to suit your needs.

If you want to train on multiple nodes, change the `GPU_PER_NODE`, `NUM_NODES`, `MASTER_ADDR`, `MASTER_PORT`, `NODE_RANK`, `WORLD_SIZE` in the script accordingly.
Alternatively, you can use elastic training. See [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) for more details.

2. After training, you will find separated trace files in the current directory. The trace files are named as `benchmark-data-{}-pipeline-{}-tensor-{}.json`, where `{}` is the rank number. Now we should aggregate the trace files into a single trace file:

```bash
python tools/aggregate.py --b trace_output --output benchmark.json
```

3. You can visualize the trace file using Chrome Tracing (or Perfetto UI). Open the trace file in Chrome Tracing by navigating to `chrome://tracing` in your browser (or https://ui.perfetto.dev/). Now you can explore the trace data, zoom in on specific events, and analyze the performance characteristics of your distributed training run.

![](images/trace1.png)

![](images/trace2.png)

4. To illustrate the detection algorithm, we can manually inject a fault into the training process. We provide a script `scripts/gpu_control.sh` to simulate a GPU downclocking.

    1. Run the script to inject a fault into the training process:

    ```bash
    # Inject a fault into GPU 0, downclocking it to 900MHz
    bash scripts/gpu_control.sh limit 0 900
    ```

    2. Run the training script. Then aggregate the trace files as described above, but with an additional command line argument to enable the detection algorithm:
    
    ```bash
    python tools/aggregate.py \
        -b . \ # Equivalent to --bench-dir
        -d # Enable the detection algorithm, Equivalent to --detect
    ```
    We can see output indicating that GPU 0 may be abnormal.

    ![1](images/result.png)

## MegaScope

### 1. Launch the Service
First, start the backend and frontend servers.

**Backend (Megatron)**: For inference mode, run the text generation server script, pointing it to your model and tokenizer paths, **and make sure to turn on the switch `--enable-ws-server` in the argument**.
```bash
bash examples/inference/a_text_generation_server_bash_script.sh /path/to/model /path/to/tokenizer
```
For example
```bash
bash examples/inference/llama_mistral/run_text_generation_llama3.sh /gfshome/llama3-ckpts/Meta-Llama-3-8B-Instruct-megatron-core-v0.12.0-TP1PP1 /root/llama3-ckpts/Meta-Llama-3-8B-Instruct
```
For training mode, run the training script, **and add `--training-ws-port XXX` (e.g. `--training-ws-port 5000`) to the argument**. A typical command is
```bash
bash a_pretrain_script.sh $RANK
```
For example
```bash
bash scripts/pretrain_gpt.sh 0
```

**Frontend (Vue)**: Navigate to the frontend directory and start the development server.
```bash
cd tools/visualization/transformer-visualize
npm run dev
```
After launching both, open your browser to the specified address (usually http://localhost:5173). You will see the main interface.

### 2. Generating Text and Visualizing Intermediate States
In the input prompts area, enter one or more prompts. Each text box represents a separate batch, allowing for parallel processing and comparison.
![](images/prompts.jpg)

In the control panel, set the desired number of tokens to generate. Also enable or disable the real-time display of specific internal states, such as QKV vectors and MLP outputs. This helps manage performance and focus on relevant data. The filter expressions of vectors can be customized by the input box below.
![](images/controls.jpg)

After starting generation, the visualization results will update token-by-token. In the first tab, the intermediate vector heatmaps are displayed and the output probabilities are shown in the expandable sections.
![](images/visualization.jpg)

The second tab contains attention matrices. Use the dropdown menus to select the layer and attention head you wish to inspect.
![](images/attention.jpg)

The third tab is the PCA dimensionality reduction feature where you can visually inspect the clustering of tokens and understand how the model groups similar concepts. The displayed layer can also be selected.
![](images/pca.jpg)

### 3. Injecting Model Perturbations
The expandable perturbation control panel can introduce controlled noise into the model's forward pass. Each kind of perturbation has an independent switch, controlling the noise type and intensity.

The currently supported noise types include:
- Additive Gaussian Noise (noise1): output = input + N(0, coef²), where N is a random value from a Gaussian (normal) distribution with mean 0.
- Multiplicative Uniform Noise (noise2): output = input * U(1 - val, 1 + val), where U is a random value from a uniform distribution.
![](images/perturbation.jpg)

### 4. Support for Training Process
Similar visualization support is provided during the training process. The overall control is the same, and the training process will be controlled on the frontend page. Critical intermediate results and perturbations are supported in training.
![](images/training.jpg)

## MegaDPP

### Environment Configuration

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

- The Python environment in the image automatically includes almost all of the required packages. To install additional required packages, run

```bash
pip install -r requirements/requirements.txt
```

- Install infiniband prerequisites

```bash
bash scripts/prerequisite.sh
```

- Build the `shm_tensor_new_rdma` (for multinode) and `shm_tensor_new_rdma_pre_alloc` modules.

```bash
cd megatron/shm_tensor_new_rdma
pip install -e .
```

```bash
cd megatron/shm_tensor_new_rdma_pre_alloc
pip install -e .
```

### Run

#### Dataset Preparation

The dataset preparation step follows largely from the Megatron framework.

First, prepare your dataset in the following `.json` format with one sample per line

```json
{"src": "bloomberg", "text": "BRIEF-Coach Inc launches tender offer to acquire Kate Spade & Co for $18.50 per share in cash. May 26 (Reuters) - Coach Inc: * Coach Inc launches tender offer to acquire Kate Spade & Company for $18.50 per share in cash * Coach Inc launches tender offer to acquire kate spade & company for $18.50 per share in cash * Coach Inc - tender offer will expire at 11:59 P.M. Edt on June 23, 2017, unless extended * Coach Inc - Chelsea Merger Sub Inc, has commenced a tender offer for all of outstanding shares of common stock, par value $1.00 per share, of Kate Spade & Company Source text for Eikon: Further company coverage: May 26 (Reuters) - Coach Inc: * Coach Inc launches tender offer to acquire Kate Spade & Company for $18.50 per share in cash * Coach Inc launches tender offer to acquire kate spade & company for $18.50 per share in cash * Coach Inc - tender offer will expire at 11:59 P.M. Edt on June 23, 2017, unless extended * Coach Inc - Chelsea Merger Sub Inc, has commenced a tender offer for all of outstanding shares of common stock, par value $1.00 per share, of Kate Spade & Company Source text for Eikon: Further company coverage:", "type": "Eng", "id": "0", "title": "BRIEF-Coach Inc launches tender offer to acquire Kate Spade & Co for $18.50 per share in cash. "}
{"src": "bloomberg", "text": "Var Energi agrees to buy Exxonmobil's Norway assets for $4.5 bln. MILAN, Sept 26 (Reuters) - Var Energi AS, the Norwegian oil and gas group 69.6% owned by Italian major Eni, has agreed to buy the Norwegian upstream assets of ExxonMobil for $4.5 billion. The deal is expected to be completed in the final quarter of this year, Var Energi said on Thursday. Reporting by Stephen Jewkes; editing by Francesca Landini MILAN, Sept 26 (Reuters) - Var Energi AS, the Norwegian oil and gas group 69.6% owned by Italian major Eni, has agreed to buy the Norwegian upstream assets of ExxonMobil for $4.5 billion. The deal is expected to be completed in the final quarter of this year, Var Energi said on Thursday. Reporting by Stephen Jewkes; editing by Francesca Landini", "type": "Eng", "id": "1", "title": "Var Energi agrees to buy Exxonmobil's Norway assets for $4.5 bln. "}
{"src": "bloomberg", "text": "Trump says 'incorrect' he is willing to meet Iran with 'no conditions'. WASHINGTON (Reuters) - U.S. President Donald Trump on Sunday appeared to play down the chances that he might be willing to meet with Iranian officials, saying reports that he would do so without conditions were not accurate. \u201cThe Fake News is saying that I am willing to meet with Iran, \u2018No Conditions.\u2019 That is an incorrect statement (as usual!),\u201d Trump said on Twitter. In fact, as recently as on Sept. 10, U.S. Secretary of State Mike Pompeo said \u201cHe (Trump) is prepared to meet with no preconditions.\u201d Reporting By Arshad Mohammed; Editing by Shri Navaratnam WASHINGTON (Reuters) - U.S. President Donald Trump on Sunday appeared to play down the chances that he might be willing to meet with Iranian officials, saying reports that he would do so without conditions were not accurate. \u201cThe Fake News is saying that I am willing to meet with Iran, \u2018No Conditions.\u2019 That is an incorrect statement (as usual!),\u201d Trump said on Twitter. In fact, as recently as on Sept. 10, U.S. Secretary of State Mike Pompeo said \u201cHe (Trump) is prepared to meet with no preconditions.\u201d Reporting By Arshad Mohammed; Editing by Shri Navaratnam", "type": "Eng", "id": "2", "title": "Trump says 'incorrect' he is willing to meet Iran with 'no conditions'. "}
```
note that we have provided a sample dataset under `datasets/gpt/` and `datasets/bert/`.

Then, prepare the vocab file (gpt and bert) and the merges file (gpt-only). We have provided it in the respective directories.

For bert, run the following
```bash
cd datasets
python ../tools/preprocess_data.py \
       --input ../datasets/bert/dataset.json \
       --output-prefix bert \
       --vocab-file ../datasets/bert/vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences \
       --workers $(nproc)
```
where the paths can be changed according to the location of your files and the place where you want the generated files to be.

For GPT, run the following
```bash
cd datasets
python ../tools/preprocess_data.py \
       --input ../datasets/gpt/dataset.json \
       --output-prefix gpt \
       --vocab-file ../datasets/gpt/vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ../datasets/gpt/merges.txt \
       --append-eod \
       --workers $(nproc)
```

For other models, please refer to `nvidia/megatron` for the corresponding datasets.

#### Single Node Distributed Training
To run distributed training on a single node, go to the project root directory and run

```bash
bash scripts/run_single_gpt.sh
```

for GPT and

```bash
bash scripts/run_single_bert.sh
```

for bert.

The `scripts/run_single_<model>.sh` files have the following structure:

- Parameters include `pipeline_parallel`, `model_chunks` and `tensor_parallel`
- The `virtual_stage_layer` parameter specifies how many layers there are in a single virtual pipeline stage. It is calculated as
$$
\frac{\text{total layer of model}}{\text{pipeline parallel}\times\text{model chunks}}
$$
where total layer is set under `examples/` the corresponding model.
- It gets the IP address of the pod and writes it to the shell script.
- Finally it runs the shell script under the corresponding model under `examples/`

There are also several critical parameters in `examples/gpt3/train_gpt3_175b_distributed.sh` (bert model under the corresponding `bert/` directory)

- `--use-dpp` switches to DPP algorithm
- `--workload` specifies the workload of each single thread, and hence determines the number of threads used in P2P communication
- `--num-gpus` specifies the number of GPUs on the current node (single node training)
- Other critical parameters include the number of layers of the model, the global batch size and the sequence length
- Note that currently the global batch size value is 16 and is static in `scripts/run_single_<model>.sh`. It needs to simultaneously modify `scripts/run_single_<model>.sh` if adjusting the layers.

For the remaining models, you can either directly run
```bash
bash examples/<model>/<train_file>.sh
```
or write a file similar to `scripts/run_{single,master,worker}_<model>.sh` that sets up configurations and runs the shell under `examples/`

#### Multinode Distributed Training
To run distributed training on multiple nodes, go to the root directory. First run

```bash
bash scripts/run_master_<model>.sh
```

and then start another pod and run

```bash
bash scripts/run_worker_<model>.sh
```

The `scripts/run_master_<model>.sh` has the following parameters

- Similar to `scripts/run_single_<model>.sh`, we have `pipeline_parallel`, `model_chunks` and `tensor_parallel`
- It writes the master pod IP to `examples/gpt3/train_gpt3_175b_distributed_master.sh` and to `train_gpt3_175b_distributed_worker.sh` (bert in the corresponding directory)
- Set the number of nodes to be 2 and master node has rank 0
- Starts the shell under `examples`

and `scripts/run_worker_<model>.sh` does the following
- Set the number of nodes to be 2 and the worker node has rank 1
- Starts the shell under `examples`

The `examples/gpt3/train_gpt3_175b_distributed_master.sh` and `examples/gpt3/train_gpt3_175b_distributed_worker.sh` are similar to the single node version, except that the `--node-ips` is mandatory, which is the infiniband IPs of the pods in the order of their GPU ranks. And also the `--multi-node` flag should be turned on.

### Profiling

Each run will generate a trace dir in `benchmark`. Go to the `tools/profiling` directory and run

```
python tools/aggregate.py --benchmark_dir benchmark/your-benchmark-dir
```

in the root dir to produce an aggregated trace file.

## MegaFBD
### Install

- Install infiniband prerequisites. 

- Build the RDMA C++ extension modules: `shm_tensor_new_rdma` (for multinode) and `shm_tensor_new_rdma_pre_alloc` module. 

Just follow above installation instructions.


### Run example

$\quad$ To run distributed training on a single node, go to the project root directory and run

```bash
bash scripts/pretrain_gpt.sh $RANK
```

Here `scripts/pretrain_gpt.sh` is an example pretraining `Bash` script. 

There are two extra options: `--forward-backward-disaggregating` and `--ignore-forward-tensor-parallel` in `TRAINING_ARGS`.

- `--forward-backward-disaggregating`


  Splits each rank into two: one for forward pass and one for backward pass. After doing this, your DP will be halved. Make sure your DP is even before adding this option.

- `--ignore-forward-tensor-parallel`

  Enables merging forward ranks within the same TP group. After doing this, your number of ranks will be multiplied by $\frac{TP+1}{2TP}$. Be sure you are using the correct number of ranks.

Currently Context Parallel and Expert parallel are not supported. `--transformer-impl` should be `local`.

# 🛠️ Security Policy

If you find a security issue with our project, report the vulnerability privately to [OpenSQZ](mailto:ospo@sqz.ac.cn). It is critical to avoid public disclosure.

An overview of the vulnerability handling process is:

- The reporter reports the vulnerability privately to [OpenSQZ](mailto:ospo@sqz.ac.cn).

- The appropriate project's security team works privately with the reporter to resolve the vulnerability.

- The project creates a new release of the package the vulnerability affects to deliver its fix.

- The project publicly announces the vulnerability and describes how to apply the fix.

# Contributing
Contributions and collaborations are welcome and highly appreciated. Check out the [Contributor Guide](https://github.com/OpenSQZ/MegatronApp/blob/main/CONTRIBUTING.md) and get involved.

# 💡 License
This project is licensed under the Apache 2.0 License, see the LICENSE file for details. 

# 🌐 Community and Support
Use WeChat to scan below QR code.

![](images/code.png)
