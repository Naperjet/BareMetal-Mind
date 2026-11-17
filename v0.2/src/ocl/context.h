#pragma once

#include "ocl.h"

#include <cstddef>
#include <string>
#include <vector>

namespace p1::ocl
{
  class Context
  {
    public:
      Context();
      ~Context();

      Context(const Context&) = delete;
      Context& operator=(const Context&) = delete;
      Context(Context&& other) noexcept;
      Context& operator=(Context&& other) noexcept;

      static bool available();

      cl_context handle() const 
      { 
        return context_; 
      }

      cl_command_queue queue() const
      {
        return queue_;
      }

      cl_device_id device() const
      {
        return device_;
      }


      std::string platform_name() const;
      std::string device_name() const;
      std::string device_version() const;
      std::string driver_version() const;
      cl_uint compute_units() const;
      std::size_t max_work_group_size() const;

    private:
      cl_platform_id platform_ = nullptr;
      cl_device_id device_ = nullptr;
      cl_context context_ = nullptr;
      cl_command_queue queue_ = nullptr
  };

}
