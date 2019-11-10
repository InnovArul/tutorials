import tensorflow as tf
import _requ_grad
# tf.enable_eager_execution()

requ_module = tf.load_op_library("build/librequ.so")

sess = tf.Session('')

with sess.as_default():
    x = tf.random.normal(shape=[3, 4])
    print_op = tf.print(x)
    with tf.control_dependencies([print_op]):
        requ_out = requ_module.requ(x)
        print_op = tf.print(requ_out)
        with tf.control_dependencies([print_op]):
            requ_grad_out = tf.gradients(requ_out, x)
            print_op = tf.print(requ_grad_out)

    sess.run(print_op)