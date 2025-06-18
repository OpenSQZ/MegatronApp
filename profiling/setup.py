from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='shm_benchmark',
    ext_modules=[
        CppExtension(
            name='shm_benchmark',
            sources=['shm_benchmark.cpp'],
            extra_compile_args=['-std=c++17', '-fPIC'],
            extra_link_args=[],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)