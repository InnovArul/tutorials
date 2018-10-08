from __future__ import division
from __future__ import print_function

import argparse
import torch

from torch.autograd import Variable, gradcheck

parser = argparse.ArgumentParser()
parser.add_argument('example', choices=['py', 'cpp', 'cuda'])
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('-f', '--features', type=int, default=7)
parser.add_argument('-s', '--state-size', type=int, default=5)
parser.add_argument('-c', '--cuda', action='store_true')
options = parser.parse_args()

if options.example == 'py':
    from python.requ import ReQU
elif options.example == 'cpp':
    from cpp.lltm import LLTMFunction
else:
    from cuda.lltm import LLTMFunction
    options.cuda = True

X = torch.randn(options.batch_size, options.batch_size, options.features, options.features)

variables = [X]

for i, var in enumerate(variables):
    if options.cuda:
        var = var.cuda()
    variables[i] = var.double()
    variables[i].requires_grad=True

if gradcheck(ReQU(), variables):
    print('Ok')
