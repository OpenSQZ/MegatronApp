<div align="center">

**带有动态流水线并行 (Dynamic Pipeline Parallel) 的 Megatron 分布式训练框架**

**📢 公告**

> Megatron DPP 当前支持多机训练以及张量并行

**🌍 选择语言**

[英文](README.md)
[中文](README_zh.md)

</div>

# 🔥 Demo

Insert a product demo.

It is recommended that the demo is within 2mins.

# 🌟 Overview

An integration of a dynamic pipeline-parallel algorithm to the Megatron distributed training framework.

# ✨ Core Features

- Supports a dynamic pipeline-parallel algorithm that selects the next microbatch to compute via a greedy rule.

- Supports tensor transfer between GPUs locally via shared memory.

- Supports a flag `--use-dpp` to switch between dynamic algorithm and original pipeline algorithm.

- Supports a flag `--node-ips` to pass in the infiniband IPs of different nodes.

# 🗺️ Tech Architecture / Project Diagram / Workflow

Illustrate the key technical points with technical architecture, workflow and so on.

# 📥 Supported Data Sources & Language Models

| Data Sources You Can Add | Supported Language Models |
| ------------------------ | ------------------------- |
| xxx                      | xxx                       |
| xxx                      | xxx                       |

# 🚀 Quickstart

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

- Build the `shm_tensor_new_rdma_multithread` module.

```bash
cd megatron/shm_tensor_new_rdma_multithread
pip install -e .
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

note that remember to change the flag `--node-ips` into the correct infiniband IPs.

## Profiling

$\quad$ Each run will generate a trace dir in `benchmark`. Run

```python
python aggregate.py --benchmark_dir benchmark/your-benchmark-dir
```

in the root dir to produce an aggregated trace file.

# 🛠️ Security Policy

If you find a security issue with our project, report the vulnerability privately to [OpenSQZ](mailto:ospo@sqz.ac.cn). It is critical to avoid public disclosure.

An overview of the vulnerability handling process is:

- The reporter reports the vulnerability privately to [OpenSQZ](mailto:ospo@sqz.ac.cn).

- The appropriate project's security team works privately with the reporter to resolve the vulnerability.

- The project creates a new release of the package the vulnerability affects to deliver its fix.

- The project publicly announces the vulnerability and describes how to apply the fix.

# 🚰 Citation

If you use or extend our work, please kindly cite xxx.

# Contributing

Contributions and collaborations are welcome and highly appreciated. Check out the [contributor guide]() and get involved.

# 💡 License

This project is licensed under the Apache 2.0 License, see the LICENSE file for details.

# 🌐 Community and Support

Provide contact information, including

- Email(user/dev email addresses, with self-subscribe service)
- Discord / Slack
- WeChat / DingTalk
- Twitter / Zhihu...
