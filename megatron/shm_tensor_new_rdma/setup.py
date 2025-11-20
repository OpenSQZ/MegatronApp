from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, include_paths
import torch

abi_flag = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
if abi_flag is None:
    abi_flag = 1

abi_macro = f"-D_GLIBCXX_USE_CXX11_ABI={int(abi_flag)}"

common_extra_compile_args = [
    "-fPIC",
    "-std=c++17",
    abi_macro,
    "-I/usr/local/cuda/include",
]

common_extra_link_args = [
    "-Wl,-rpath,$ORIGIN",
    "-L/usr/local/cuda/lib64",
    "-lcudart",
]


setup(
    name="shm_tensor_new_rdma",
    ext_modules=[
        CppExtension(
            name="shm_tensor_new_rdma",
            sources=["shm_tensor_new_rdma.cpp"],
            include_dirs=include_paths(),
            libraries=["rdmacm", "ibverbs", "torch", "torch_python", "c10"],
            extra_compile_args=common_extra_compile_args,
            extra_link_args=common_extra_link_args,
        ),
        CppExtension(
            name="shm_tensor_new_rdma_pre_alloc",
            sources=["shm_tensor_new_rdma_pre_alloc.cpp"],
            include_dirs=include_paths(),
            libraries=["rdmacm", "ibverbs", "torch", "torch_python", "c10"],
            extra_compile_args=common_extra_compile_args,
            extra_link_args=common_extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=[],
)
