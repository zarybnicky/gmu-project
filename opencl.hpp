#ifndef OPENCL_H
#define OPENCL_H

#include "include.hpp"
#include "kernel.h"

class OpenCL {
public:
  static cl::Program clprogram;
  static cl::CommandQueue clqueue;
  static cl::Context clcontext;
  static void initialize_OpenCL();
};
#endif
