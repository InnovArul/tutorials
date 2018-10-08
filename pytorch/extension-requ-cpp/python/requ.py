import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)

class ReQU(torch.nn.Module):
    def __init__(self):
        super(ReQU, self).__init__()

    def forward(self, input):
        input = F.relu(input)
        return torch.pow(input, 2)