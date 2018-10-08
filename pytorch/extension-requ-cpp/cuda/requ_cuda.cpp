#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

at::Tensor requ_cuda_forward(at::Tensor input);
at::Tensor requ_cuda_backward(at::Tensor input, at::Tensor output, at::Tensor grad_output);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor requ_forward(at::Tensor input) {
  CHECK_INPUT(input);
  return requ_cuda_forward(input);
}

at::Tensor requ_backward(at::Tensor input, at::Tensor output, at::Tensor grad_output) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(grad_output);

  return requ_cuda_backward(input, output, grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &requ_forward, "LLTM forward (CUDA)");
  m.def("backward", &requ_backward, "LLTM backward (CUDA)");
}
