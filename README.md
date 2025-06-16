# MegatronDPP (Megatron with Dynamic Pipeline-Parallel)

$\quad$ An integration of a dynamic pipeline-parallel algorithm to the Megatron distributed training framework.

- Supports a dynamic pipeline-parallel algorithm that selects the next microbatch to compute via a greedy rule.

- Supports tensor transfer between GPUs locally via shared memory.

- Supports a flag `--use-dpp` to switch between dynamic algorithm and original pipeline algorithm.

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

$\quad$ To run distributed training on a single node, go to the project root directory and run

```bash
bash run_single.sh
```

note that there is a flag `--use-dpp` in `TRAINING_ARGS`. Remove it to use original pipeline algorithm.

$\quad$ To run distributed training on multi nodes, go to the root directory. First run

```bash
bash run_master.sh
```

and then start another pod and run

```bash
bash run_worker.sh
```

## Profiling

$\quad$ Each run will generate a trace dir in `benchmark`. Run

```python
python aggregate.py --benchmark_dir benchmark/your-benchmark-dir
```

in the root dir to produce an aggregated trace file.
