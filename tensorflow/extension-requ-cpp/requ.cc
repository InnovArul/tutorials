// references
// https://davidstutz.de/implementing-tensorflow-operations-in-c-including-gradients/


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/default/logging.h"

using namespace tensorflow;


// register the op
REGISTER_OP("Requ")
        .Input("input: float32")
        .Output("output: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
            shape_inference::ShapeHandle input_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

            shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
            shape_inference::DimensionHandle input_cols = c->Dim(input_shape, 1);

            c->set_output(0, c->Matrix(input_rows, input_cols));
            return Status::OK();
        });


// implementation of the op
class RequOp : public OpKernel {
public:
    // constructor
    explicit RequOp(OpKernelConstruction* context) : OpKernel(context) {}

    // override compute function
    void Compute(OpKernelContext* context) override {
        // basic checks
        DCHECK_EQ(1, context->num_inputs());

        // get input tensor
        const Tensor& input = context->input(0);
        const TensorShape& input_shape = input.shape();
        DCHECK_EQ(input_shape.dims(), 2);

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(input_shape.dim_size(0));
        output_shape.AddDim(input_shape.dim_size(1));

        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        // get corresponding eigen vectors
        auto input_tensor = input.matrix<float>();
        auto output_tensor = output->matrix<float>();

        for (int i = 0; i < input_shape.dim_size(0); i++) {
            for (int j = 0; j < input_shape.dim_size(1); j++) {
                if(input_tensor(i, j) > 0) {
                    output_tensor(i, j) = input_tensor(i, j) * input_tensor(i, j);
                }
                else {
                    output_tensor(i, j) = 0;
                }
                
            }
            
        }
        
    }
};


REGISTER_KERNEL_BUILDER(
    Name("Requ").Device(DEVICE_CPU), RequOp
);

