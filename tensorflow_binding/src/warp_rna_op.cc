#ifdef WARPRNA_ENABLE_GPU
#define EIGEN_USE_GPU
#include <cuda.h>
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "core.h"

namespace tf = tensorflow;


REGISTER_OP("WarpRNA")
    .Input("log_probs: float32")
    .Input("labels: int32")
    .Input("input_lengths: int32")
    .Input("label_lengths: int32")
    .Input("min_u: int32")
    .Attr("blank_label: int = 0")
    .Output("costs: float32")
    .Output("grads: float32")
     .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
         c->set_output(0, c->Vector(c->Dim(c->input(1), 0)));
         c->set_output(1, c->input(0));
         return tf::Status::OK();
     });


namespace warp_rna {

const char* rnaGetStatusString(rnaStatus_t status) {
    switch (status) {
    case RNA_STATUS_SUCCESS:
        return "no error";
    case RNA_STATUS_WARP_FAILED:
        return "warp failed";
    case RNA_STATUS_GRADS_BLANK_FAILED:
        return "kernel launch failed: grads of blank";
    case RNA_STATUS_GRADS_LABEL_FAILED:
        return "kernel launch failed: grads of labels";
    case RNA_STATUS_COSTS_FAILED:
        return "kernel launch failed: costs";
    default:
        return "unknown error";

    }

}

// Currently there is only a GPU version.

#ifdef WARPRNA_ENABLE_GPU

class WarpRNAOpGPU : public tf::OpKernel {
  public:
    explicit WarpRNAOpGPU(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_label", &blank_label_));
    }

    void Compute(tf::OpKernelContext* ctx) {
        const tf::Tensor* log_probs = nullptr;
        const tf::Tensor* labels = nullptr;
        const tf::Tensor* label_lengths = nullptr;
        const tf::Tensor* input_lengths = nullptr;
        const tf::Tensor* min_u = nullptr;
        OP_REQUIRES_OK(ctx, ctx->input("log_probs", &log_probs));
        OP_REQUIRES_OK(ctx, ctx->input("labels", &labels));
        OP_REQUIRES_OK(ctx, ctx->input("label_lengths", &label_lengths));
        OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lengths));
        // We need the minimum label-lengths for allocation, thus we simply pass it here.
        OP_REQUIRES_OK(ctx, ctx->input("min_u", &min_u));


        OP_REQUIRES(ctx, log_probs->shape().dims() == 4,
                    tf::errors::InvalidArgument("log_probs is not a 4-Tensor"));
        OP_REQUIRES(ctx, labels->shape().dims() == 2,
                    tf::errors::InvalidArgument("labels is not a 2-Tensor"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(label_lengths->shape()),
                     tf::errors::InvalidArgument("label_lengths is not a vector"));
        OP_REQUIRES(ctx, tf::TensorShapeUtils::IsVector(input_lengths->shape()),
                     tf::errors::InvalidArgument("input_lengths is not a vector"));
        OP_REQUIRES(ctx, min_u->shape().dims() == 0,
                    tf::errors::InvalidArgument("min_u is not a scalar."));

        const auto& log_probs_shape = log_probs->shape();
        const auto batch_size = log_probs_shape.dim_size(0);
        const auto max_time = log_probs_shape.dim_size(1);
        const auto max_u = log_probs_shape.dim_size(2);
        const auto num_classes_raw = log_probs_shape.dim_size(3);

        auto log_probs_t = log_probs->tensor<float, 4>();
        auto labels_t = labels->tensor<int32_t, 2>();
        const Eigen::Tensor<int, 0, Eigen::RowMajor> min_u_t = min_u->tensor<int32_t, 0>();
        int max_s = max_time - min_u_t() + 1;

        OP_REQUIRES(
                ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
                tf::errors::InvalidArgument("num_classes cannot exceed max int"));

        OP_REQUIRES(
                ctx, batch_size == input_lengths->dim_size(0),
                tf::errors::InvalidArgument("len(input_lengths) != batch_size.  ",
                                            "len(input_length):  ", input_lengths->dim_size(0),
                                            " batch_size: ", batch_size));
        auto input_lengths_t = input_lengths->vec<int>();

        OP_REQUIRES(
                ctx, batch_size == label_lengths->dim_size(0),
                tf::errors::InvalidArgument("len(label_lengths) != batch_size.  ",
                                            "len(label_length):  ", label_lengths->dim_size(0),
                                            " batch_size: ", batch_size));
        auto label_lengths_t = label_lengths->vec<int>();


        tf::Tensor* costs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("costs", input_lengths->shape(), &costs));
        auto costs_t = costs->vec<float>();

        tf::Tensor* grads = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("grads", log_probs->shape(), &grads));
        auto grads_t = grads->tensor<float, 4>();


        auto counts_shape = tf::TensorShape{batch_size, max_u* 2};
        tf::Tensor counts;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_UINT32, counts_shape, &counts));
        cudaMemset(counts.data(), 0, batch_size*max_u*2*sizeof(uint32_t));
        auto counts_t = counts.tensor<unsigned int, 2>();

        // for both alphas and betas
        auto buffer_shape = tf::TensorShape{batch_size, max_s, max_u};
        tf::Tensor alphas, betas;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_FLOAT, buffer_shape, &alphas));
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(tf::DT_FLOAT, buffer_shape, &betas));
        auto alphas_t = alphas.tensor<float, 3>();
        auto betas_t = betas.tensor<float, 3>();

        auto cuda_stream = ctx->eigen_gpu_device().stream();
        auto rna_status = run_warp_rna(cuda_stream, counts_t.data(), alphas_t.data(), betas_t.data(), labels_t.data(),
                                  log_probs_t.data(), grads_t.data(), costs_t.data(),
                                  input_lengths_t.data(), label_lengths_t.data(),
                                  batch_size, max_time, max_s, max_u, num_classes_raw, blank_label_);
        OP_REQUIRES(ctx, rna_status == RNA_STATUS_SUCCESS,
                    tf::errors::Internal("warp_rna error in compute_rna_loss:",
                                         rnaGetStatusString(rna_status)));
    }

  private:
      int blank_label_;
};

REGISTER_KERNEL_BUILDER(Name("WarpRNA").Device(::tensorflow::DEVICE_GPU)
                                       .HostMemory("min_u"),
                        WarpRNAOpGPU);
#undef EIGEN_USE_GPU
#endif

}
