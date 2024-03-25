
import distutils.spawn
import subprocess

from .builder import CUDAOpBuilder


class AsyncCopierBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_ASYNC_COPIER"
    NAME = "async_copier"

    def __init__(self):
        super().__init__(name=self.NAME)

    def absolute_name(self):
        return f'deepspeed.ops.async_copier.{self.NAME}_op'

    def sources(self):
        return ['csrc/async_copier/async_copier.cpp', 'csrc/async_copier/py_async_copier.cpp']

    def include_paths(self):
        return ['csrc/async_copier', 'csrc/includes']

    def cxx_args(self):
        args = super().cxx_args() + ['-fopenmp']
        return args + self.version_dependent_macros()

    def nvcc_args(self):
        # nvcc_flags = ['-O3', '-std=c++14', '-static-libstdc++'] + self.version_dependent_macros()
        nvcc_flags = ['-O3', '-lineinfo'] + self.version_dependent_macros()
        return nvcc_flags
