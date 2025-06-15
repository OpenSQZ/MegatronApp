# MegatronAPP

$\quad$ An integration of an adaptive pipeline-parallel algorithm to the Megatron distributive training framework.

- Supports an adaptive pipeline-parallel algorithm that selects the next microbatch to compute via a greedy rule.

- Supports tensor transfer between GPUs locally via shared memory.

- Supports a flag `--use-app` to switch between adaptive algorithm and original pipeline algorithm.

## Environment Configuration

- The following is the pod configuration.

```yaml
ContainerImage: ngc.nju.edu.cn/nvidia/pytorch:25.03-py3
GPU: RTX4090

Limits:
  CPU: 16
  memory: 200Gi
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

- Build the `shm_tensor_new_rdma` module.

```bash
cd megatron
python setup.py install
```

## Run

$\quad$ To run distributed training, go to the project root directory and run

```bash
bash examples examples/gpt3/train_gpt3_175b_distributed.sh
```

note that there is a flag `--use-app` in `TRAINING_ARGS`. Remove it to use original pipeline algorithm.
