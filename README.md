# MegatronAPP

$\quad$ An integration of an adaptive pipeline-parallel algorithm to the Megatron distributive training framework.

- Supports an adaptive pipeline-parallel algorithm that selects the next microbatch to compute via a greedy rule.

- Supports tensor transfer between GPUs locally via shared memory, and between GPUs on different nodes through Remote Direct Memory Access (RDMA).

## Environment Configuration

- Create $2$ pods to simulate $2$ nodes with the following configuration

```yaml
ContainerImage: ngc.nju.edu.cn/nvidia/pytorch:23.04-py3
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
python setup.py install
```

## Run Single Node

- Run the following script to start Megatron on a single node with shared memory transfer. Remembor to change the master ip address in `examples/megatron4.0/pretrain_gpt_distributed_small.sh` to the ip of the pod

```bash
bash run_single.sh --pipeline_parallel <pipeline_parallel_size> --model_chunks <number_of_model_chunks>
```

- It will produce a `trace_pipeline_<pipeline_parallel_size>_model_chunks_<number_of_model_chunks>.txt` that stores the profile of the program.

- To visualize the pipeline, first run

```python
python process_trace.py --pipeline_parallel <pipeline_parallel_size> --model_chunks <number_of_model_chunks>
```

and then run

```python
python visualize_trace.py --pipeline_parallel <pipeline_parallel_size> --model_chunks <number_of_model_chunks>
```

the trace will be stored in `trace_pipeline_<pipeline_parallel>_model_chunks_<number_of_model_chunks>.xlsx`. It looks like

![pipeline](pipeline_sample.png)

which shows the computation order of the pipeline.

## Run Multiple Nodes

- Attach to a pod (or use command-line) and run:

```bash
bash run_master.sh
```

and wait until it fully starts. Then run

```bash
bash run_worker.sh
```

using command-line on another pod. Remember to edit the `shm_tensor_new_rdma.cpp` file to set the infiniband ip to the real pod infiniband ip.
