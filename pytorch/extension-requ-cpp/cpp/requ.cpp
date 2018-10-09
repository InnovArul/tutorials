#include <torch/extension.h>
#include <vector>
#include <iostream>
using namespace std;


at::Tensor requ_forward(at::Tensor input)
{
  auto relu_input = at::relu(input);
  auto output = at::pow(relu_input, 2);
  return output;
}

at::Tensor requ_backward(at::Tensor input,
                                      at::Tensor output, 
                                      at::Tensor grad_output) 
{
  auto relu_input = at::relu(input) * 2;
  auto d_input = relu_input * grad_output;
  return d_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &requ_forward, "ReQU forward");
  m.def("backward", &requ_backward, "ReQU backward");
}
