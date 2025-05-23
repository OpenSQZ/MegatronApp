# MegatronAPP

$\quad$ An integration of an adaptive pipeline-parallel algorithm to the Megatron distributive training framework.

- Supports an adaptive pipeline-parallel algorithm that selects the next microbatch to compute via a greedy rule.

- Supports tensor transfer between GPUs locally via shared memory, and between GPUs on different nodes through Remote Direct Memory Access (RDMA).

## Environment Configuration

- Create $2$ pods to simulate $2$ nodes with the following configuration

```yaml
ContainerImage: harbor-local.ai.iiis.co/llm-course/comfyui:v1
GPU: RTX4090

NVMEStorage: 100G
Limits:
  CPU: 16
  memory: 200Gi
  GPU: 4
UseShm: True
ShmSize: 16Gi

UseIB: true
```

note that since attaching Visual Studio Code to multiple pods risks overwriting user profile, which results in being unable to open files/directories, it is recommended to attach to $1$ of the pods and operate on the other using command-line. Of course, you can operate on both via terminal.

- On pod $1$, create conda environment. Here if miniconda is installed under the `home` directory, then it is automatically shared across pods so we only need to set up the environment once

```bash
conda create -n megatron_app python=3.10
conda activate megatron_app
```

- Install torch, torchvision, torchaudio, pybind11

```bash
pip install torch==2.4.0 torchvision torchaudio pybind11
```

- Build `transformer_engine`.

```bash
cd TransformerEngine
MAX_JOBS=8 pip install .
```

note that this may take a while to build transformer_engine and flash_attn.

- Build `apex`.

```bash
cd apex
python setup.py install --cpp_ext --cuda_ext
```

and it might take a while, and it might be necessary to comment the line `check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)` in `apex/setup.py`.

```python
if "--cuda_ext" in sys.argv:
    sys.argv.remove("--cuda_ext")
    raise_if_cuda_home_none("--cuda_ext")
    # check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)
```

- Install remaining packages

```bash
pip install six regex nltk
```

- Install infiniband prerequisites

```bash
bash prerequisites.sh
```

- Build the `shm_tensor_new_rdma` module.

```bash
python setup.py install
```

## Run Single Node

- Run the following script to start Megatron on a single node with shared memory transfer.

```bash
bash run_single.sh
```

## Run Multiple Nodes

- Attach to a pod (or use command-line) and run:

```bash
bash run_master.sh
```

and wait until it fully starts. Then run

```bash
bash run_worker.sh
```

using command-line on another pod.
