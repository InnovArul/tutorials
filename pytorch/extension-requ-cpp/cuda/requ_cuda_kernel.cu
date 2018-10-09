#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

namespace {

template <typename scalar_t>
__global__ void requ_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output, int num_elements) {

    int blockNumber = blockIdx.x;
    int threadNumber = threadIdx.x;
    int elementNumber = (blockNumber * 1024) + threadNumber;

    if(elementNumber < num_elements) {
      if(input[elementNumber] > 0) {
         output[elementNumber] = input[elementNumber] * input[elementNumber];
      }
    }
}

template <typename scalar_t>
__global__ void requ_cuda_backward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ output,
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input, int num_elements) {

      int blockNumber = blockIdx.x;
      int threadNumber = threadIdx.x;
      int elementNumber = (blockNumber * 1024) + threadNumber;

    if(elementNumber < num_elements) {
      if(input[elementNumber] > 0) {
        grad_input[elementNumber] = 2 * input[elementNumber] * grad_output[elementNumber];
      }
    }

}
} // namespace

at::Tensor requ_cuda_forward(at::Tensor input) {
  const int size = input.numel();
  const int threads = 1024;
  const int blocks = (size + threads - 1) / threads;
  auto output = at::zeros_like(input);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "requ_forward_cuda", ([&] {
    requ_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(input.data<scalar_t>(), 
                                                      output.data<scalar_t>(),
                                                      input.numel());
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
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        grad_output.data<scalar_t>(),
        grad_input.data<scalar_t>(),
        input.numel());
  }));

  return grad_input;
}
