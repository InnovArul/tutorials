#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// op registration
REGISTER_OP("RequGrad")
    .Input("grad: float32")
    .Input("input: float32")
    .Output("grad_input: float32");


// class implementation
class RequGradOp : public OpKernel {
    public:
        explicit RequGradOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            DCHECK_EQ(2, context->num_inputs());

            // get tensors
            const Tensor& grad = context->input(0);
            const Tensor& input = context->input(1);

            TensorShape input_shape = input.shape();

            Tensor* grad_input = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));

            // get eigen references
            auto grad_tensor = grad.matrix<float>();
            auto input_tensor = input.matrix<float>();
            auto grad_input_tensor = grad_input->matrix<float>();

            for (int i = 0; i < input_shape.dim_size(0); i++)
            {
                for (int j = 0; j < input_shape.dim_size(1); j++)
                {
                    if(input_tensor(i, j) > 0) {
                        grad_input_tensor(i, j) = 2 * input_tensor(i, j) * grad_tensor(i, j);
                    }
                    else{
                        grad_input_tensor(i, j) = 0;
                    }
                }
                
            }
            
        } 
};


REGISTER_KERNEL_BUILDER(Name("RequGrad").Device(DEVICE_CPU), RequGradOp);