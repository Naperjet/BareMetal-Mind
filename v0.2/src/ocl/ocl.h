#pragma once

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>

#include <stdexcept>
#include <string>

namespace p1::ocl
{
  std::string error_string(cl_int err);

  void check(cl_int err, const char* what);
}
