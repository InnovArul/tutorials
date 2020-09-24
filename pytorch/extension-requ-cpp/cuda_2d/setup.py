from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='requ_cuda_2d',
    ext_modules=[
        CUDAExtension('requ_cuda_2d', [
            'requ_cuda_2d.cpp',
            'requ_cuda_2d_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
