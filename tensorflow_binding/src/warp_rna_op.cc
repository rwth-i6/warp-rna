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

#ifdef WARPRNA_ENABLE_GPU

#define CLASS_NAME WarpRNAOpGPU
#define __WARPRNA_GPU
#include "warp_rna_op_kernel_tmpl.h"
#undef CLASS_NAME
#undef __WARPRNA_GPU

#endif

#ifdef WARPRNA_ENABLE_CPU

#define CLASS_NAME WarpRNAOpCPU
#define __WARPRNA_CPU
#include "warp_rna_op_kernel_tmpl.h"
#undef CLASS_NAME
#undef __WARPRNA_CPU

#endif

}
