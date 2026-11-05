#!/usr/bin/env python3
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

cxx_args = ['-std=c++17']

# Detect the compute capability of the GPU
gpu_arch = torch.cuda.get_device_capability(0)
compute_capability = f'{gpu_arch[0]}{gpu_arch[1]}'

nvcc_args = [
    '-gencode', f'arch=compute_{compute_capability},code=sm_{compute_capability}',
    '-gencode', 'arch=compute_61,code=compute_61'  # Keep original for backward compatibility
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
