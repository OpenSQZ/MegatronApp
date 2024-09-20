#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import setuptools
from setuptools import setup
from setuptools import find_packages
from torch.utils import cpp_extension
# from torch.utils.cpp_extension import CUDAExtension

# inc_lib = CUDAExtension(
#     "inc_c", 
#     sources=["inc/torch/inc.cc", "inc/torch/util.cc", "inc/torch/plugin.cc", "inc/torch/comm.cc", 
#              "inc/torch/msg.cc", "inc/torch/cuda_comm.cc", "inc/rdma/rdma_client.cc", 
#              "inc/rdma/rdma_server.cc", "inc/rdma/common.cc", "inc/rdma/rdma_common.cc", 
#              "inc/rdma/worker_client.cc", "inc/rdma/worker_server.cc", 
#              "inc/rdma/center_client.cc", "inc/rdma/center_server.cc"],
#     extra_compile_args=['-fopenmp', '-I/usr/local/cuda/include', '-std=c++20'], 
#     extra_link_args=['-lgomp', '-lnuma', '-L/usr/local/cuda/lib64', '-libverbs', '-pthread', '-lrdmacm', '-lcudart']
# )

setup(
    name="inc",
    version="0.0.2",
    author="Bohan Zhao",
    author_email="BohanZhaoIIIS@outlook.com",
    description="In-network Computation Framework for Multi-tenant Learning",
    url="https://github.com:ZeBraHack0/inc-torch.git",
    packages=find_packages(),
    ext_modules=[],
    cmdclass={},
    python_requires='>=3.6',
    install_requires=['numpy>=1.18']
)