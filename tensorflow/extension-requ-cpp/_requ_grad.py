"""
gradients for ReQU operation
"""

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
requ_grad_module = tf.load_op_library("build/librequ_grad.so")

@ops.RegisterGradient("Requ")
def _requ_grad_cc(op, grad):
    return requ_grad_module.requ_grad(grad, op.inputs[0])