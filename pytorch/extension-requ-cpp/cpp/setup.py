from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='requ_cpp',
    ext_modules=[
        CppExtension('requ_cpp', ['requ.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
