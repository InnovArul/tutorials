from torch.utils.cpp_extension import load
lltm_cuda = load(
    'requ_cuda', ['requ_cuda.cpp', 'requ_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)
