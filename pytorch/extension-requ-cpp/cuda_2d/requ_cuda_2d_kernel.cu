#include <torch/extension.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

namespace {

template <typename scalar_t>
__global__ void requ_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output) {

    int rowNumber = blockIdx.x;
    int columnNumber = threadIdx.x;

    if(input[rowNumber][columnNumber] > 0) {
      output[rowNumber][columnNumber] = input[rowNumber][columnNumber] * input[rowNumber][columnNumber];
    }
}

template <typename scalar_t>
__global__ void requ_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_output,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_input) {

      int rowNumber = blockIdx.x;
      int columnNumber = threadIdx.x;

      if(input[rowNumber][columnNumber] > 0) {
        grad_input[rowNumber][columnNumber] = 2 * input[rowNumber][columnNumber] * grad_output[rowNumber][columnNumber];
      }
}

} // namespace

at::Tensor requ_cuda_forward(at::Tensor input) {
  const int size = input.numel();
  const int threads = input.size(1);
  const int blocks = input.size(0);
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "requ_forward_cuda", ([&] {
    requ_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(), 
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  return output;
}

at::Tensor requ_cuda_backward(
    at::Tensor input,
    at::Tensor output,
    at::Tensor grad_output) {

  const int size = input.numel();
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  auto grad_input = at::zeros_like(grad_output);


  AT_DISPATCH_FLOATING_TYPES(input.type(), "requ_backward_cuda", ([&] {
    requ_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        grad_output.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        grad_input.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  return grad_input;
}
