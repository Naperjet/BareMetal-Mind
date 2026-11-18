#pragma once

#include "context.h"

#include <cstddef>

namespace p1::ocl
{
  class Buffer
  {
    public:
      Buffer() = default;
      Buffer(const Context& ctx, std::size_t bytes, cl_mem_flags flags, void* host_ptr = nullptr);
      ~Buffer();

      Buffer(const Buffer&) = delete;
      Buffer& operator=(const Buffer&) = delete;
      Buffer(Buffer&& other) noexcept;
      Buffer& operator=(Buffer&& other) noexcept;

      cl_mem handle() const
      {
        return mem_;
      }

    private:
      cl_mem mem_ = nullptr;
  };

}
