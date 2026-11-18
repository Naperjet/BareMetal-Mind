#pragma once

#include "buffer.h"
#include "context.h"

#include <cstddef>
#include <string>

namespace p1::ocl
{
  class Kernel
  {
    public:
      Kernel() = default;
      Kernel(const Context& ctx, const std::string& source_path, const std::string& kernel_name);
      ~Kernel();

      Kernel(const Kernel&) = delete;
      Kernel& operator=(const Kernel&) = delete;
      Kernel(Kernel&& other) noexcept;
      Kernel& operator=(Kernel&& other) noexcept;

      cl_kernel handle() const
      {
        return kernel_;
      }

      void set_arg(cl_uint index, const Buffer& buffer);
      void set_arg_local(cl_uint index, std::size_t bytes);

      template <typename T>
      void set_arg(cl_uint index, const T& value)
      {
        set_arg(index, sizeof(T), &value);
      }

      void set_arg(cl_uint index, std::size_t size, const void* value);

      void enqueue_1d(const Context& ctx, std::size_t global, std::size_t local, cl_event* event = nullptr) const;

    private:
      cl_program program_ = nullptr;
      cl_kernel kernel_ = nullptr;
  };

}
