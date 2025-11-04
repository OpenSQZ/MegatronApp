# Use Nanjing University mirrors for faster package downloads in China
FROM ngc.nju.edu.cn/nvidia/pytorch:25.04-py3
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 FORCE_CUDA=1

RUN sed -i 's@archive.ubuntu.com@mirror.nju.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirror.nju.edu.cn@g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git wget curl pkg-config \
      rdma-core libibverbs1 ibverbs-providers librdmacm1 \
      libibverbs-dev librdmacm-dev libnuma-dev libnl-3-dev libnl-route-3-dev \
      python3-dev python3-setuptools python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.pip && \
    printf "[global]\nindex-url = https://mirror.nju.edu.cn/pypi/web/simple\ntrusted-host = mirror.nju.edu.cn\n" > /root/.pip/pip.conf && \
    python3 -m pip install --no-cache-dir --upgrade pip wheel setuptools

WORKDIR /workspace/MegatronApp
# Copy your current warehouse content into the mirror here.
COPY . .

# Project pre-dependencies and two RDMA C++ extensions compilation
RUN bash scripts/prerequisite.sh
WORKDIR /workspace/MegatronApp/megatron/shm_tensor_new_rdma
RUN pip install -e .
WORKDIR /workspace/MegatronApp/megatron/shm_tensor_new_rdma_pre_alloc
RUN pip install -e .

WORKDIR /workspace/MegatronApp
RUN pip install -r requirements/requirements.txt
RUN python - <<'PY'
import importlib
for m in ['shm_tensor_new_rdma_pre_alloc','shm_tensor_new_rdma']:
    importlib.import_module(m)
print('RDMA C++ extensions OK')
PY
