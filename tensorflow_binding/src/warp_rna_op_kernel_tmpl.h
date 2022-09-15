
#ifdef __WARPRNA_CPU
#define cudaMemset memset
#endif


class CLASS_NAME : public tf::OpKernel {
  public:
    explicit CLASS_NAME(tf::OpKernelConstruction* ctx) : tf::OpKernel(ctx) {
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
        // It's actually the min(U-1) here.
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
        const auto max_time = log_probs_shape.dim_size(1); // T
        const auto max_u = log_probs_shape.dim_size(2); // U
        const auto num_classes_raw = log_probs_shape.dim_size(3); // incl blank

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
                                            "len(input_lengths):  ", input_lengths->dim_size(0),
                                            " batch_size: ", batch_size));
        auto input_lengths_t = input_lengths->vec<int>();

        OP_REQUIRES(
                ctx, batch_size == label_lengths->dim_size(0),
                tf::errors::InvalidArgument("len(label_lengths) != batch_size.  ",
                                            "len(label_lengths):  ", label_lengths->dim_size(0),
                                            " batch_size: ", batch_size));
        OP_REQUIRES(
                ctx, max_u == labels->dim_size(1) + 1,
                tf::errors::InvalidArgument("labels.shape[1] + 1 != max_u == log_probs.shape[2].  ",
                                            "labels.shape[1]:  ", labels->dim_size(1),
                                            " max_u: ", max_u));
        auto label_lengths_t = label_lengths->vec<int>();


        tf::Tensor* costs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("costs", input_lengths->shape(), &costs));
        auto costs_t = costs->vec<float>();

        tf::Tensor* grads = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("grads", log_probs->shape(), &grads));
        cudaMemset(grads->data(), 0, batch_size*max_time*max_u*num_classes_raw*sizeof(float));
        auto grads_t = grads->tensor<float, 4>();

        if(max_s <= 0) {
            cudaMemset(costs->data(), 0, batch_size*sizeof(float));
            return;
        }

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

        auto rna_status =
#ifdef __WARPRNA_CPU
            run_warp_rna_cpu(
#else
            run_warp_rna(
                ctx->eigen_gpu_device().stream(),
#endif
                counts_t.data(), alphas_t.data(), betas_t.data(), labels_t.data(),
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

#ifdef __WARPRNA_CPU
REGISTER_KERNEL_BUILDER(Name("WarpRNA").Device(::tensorflow::DEVICE_CPU),
                        CLASS_NAME);
#else
REGISTER_KERNEL_BUILDER(Name("WarpRNA").Device(::tensorflow::DEVICE_GPU)
                                       .HostMemory("min_u"),
                        CLASS_NAME);
#endif

#ifdef __WARPRNA_CPU
#undef cudaMemset
#endif

