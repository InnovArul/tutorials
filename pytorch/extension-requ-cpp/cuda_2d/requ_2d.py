import math
from torch import nn
from torch.autograd import Function
import torch

import requ_cuda_2d

torch.manual_seed(42)

class ReQUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = requ_cuda_2d.forward(input)
        variables = [input, output]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        d_input = requ_cuda_2d.backward(input, output, grad_output)
        return d_input


class ReQU(nn.Module):
    def __init__(self):
        super(ReQU, self).__init__()

    def forward(self, input):
        return ReQUFunction.apply(input)