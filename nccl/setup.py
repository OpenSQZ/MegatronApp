from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths, library_paths
import torch
import os

setup(
    name='send_nccl',
    ext_modules=[
        CppExtension(
            name='send_nccl',
            sources=['send_nccl.cpp'],
            include_dirs=[
                os.environ.get('CUDA_HOME', '/usr/local/cuda') + '/include',
                *include_paths(),
            ],
            library_dirs=[
                '/usr/local/cuda/lib64',
                *library_paths(),
            ],
            extra_compile_args=[
                "-fPIC",
                "-std=c++17",
                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            ],
            extra_link_args=[
                "-Wl,-rpath,$ORIGIN",
                "-lcudart",
                "-lnccl",
                "-lrt",
                "-lpthread",
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)