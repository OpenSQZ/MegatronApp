from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, include_paths

setup(
    name="shm_tensor_new_rdma",
    ext_modules=[
        CppExtension(
            name="shm_tensor_new_rdma",
            sources=["shm_tensor_new_rdma.cpp"],
            include_dirs=include_paths(),
            libraries=[
                "rdmacm", "ibverbs",
                "torch", "torch_python", "c10"
            ],
            extra_compile_args=[
                "-fPIC",
                "-std=c++17",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                "-I/usr/local/cuda/include"
            ],
            extra_link_args=[
                "-Wl,-rpath,$ORIGIN",
                "-L/usr/local/cuda/lib64",
                "-lcudart"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)