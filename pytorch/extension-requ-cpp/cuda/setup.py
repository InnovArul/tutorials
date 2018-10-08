from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='requ_cuda',
    ext_modules=[
        CUDAExtension('requ_cuda', [
            'requ_cuda.cpp',
            'requ_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
