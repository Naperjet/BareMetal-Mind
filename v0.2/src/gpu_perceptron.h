#pragma once

#include "ocl/buffer.h"
#include "ocl/context.h"
#include "ocl/kernel.h"
#include "perceptron.h"

#include <cstddef>
#include <vector>

namespace p1
{
  class GPUPerceptron
  {
    public:
      GPUPerceptron(const ocl::Context& ctx, std::vector<float> weights, float bias, Activation activation);

      float forward(const std::vector<float>& x) const;
      float predict(const std::vector<float>& x) const;

      void upload_input(const std::vector<float>& x) const;
      float forward_z_resident() const;
      double forward_kernel_time_ns() const;

      std::size_t n_inputs() const
      {
        return weights_.size();
      }

      const std::vector<float>& weights() const
      {
        return weights_;
      }

      float bias() const
      {
        return bias_;
      }

      Activation activation() const
      {
        return activation_;
      }

    private:
      void enqueue_kernel() const;

      const ocl::Context& ctx_;
      std::vector<float> weights_;
      float bias_;
      Activation activation_;
      std::size_t local_;
      ocl::Buffer weights_buf_;
      ocl::Buffer input_buf_;
      ocl::Buffer z_buf_;
      ocl::Kernel kernel_;
  };

}
