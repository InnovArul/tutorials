from torch.utils.cpp_extension import load
lltm_cpp = load(name="requ_cpp", sources=["requ.cpp"], verbose=True)
help(lltm_cpp)
