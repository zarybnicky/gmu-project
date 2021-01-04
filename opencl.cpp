#include "opencl.hpp"

cl::Program OpenCL::clprogram;
cl::Context OpenCL::clcontext;
cl::CommandQueue OpenCL::clqueue;

void OpenCL::initialize_OpenCL() {
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    std::cout << " No platforms found. Check OpenCL installation!\n";
    exit(1);
  }

  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if (all_devices.size() == 0) {
    std::cout << " No devices found. Check OpenCL installation!\n";
    exit(1);
  }

  cl::Device default_device = all_devices[0];
  std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
  OpenCL::clcontext = cl::Context({ default_device });

  std::string src((char *) kernel_cl, kernel_cl_len - 1);
  cl::Program::Sources sources({{ src.c_str(), src.size() }});
  OpenCL::clprogram = cl::Program(OpenCL::clcontext, sources);
  try {
    OpenCL::clprogram.build({ default_device });
  }
  catch (...) {
    cl_int buildErr = CL_SUCCESS;
    auto buildInfo = OpenCL::clprogram.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
    for (auto &pair : buildInfo) {
      std::cerr << pair.second << std::endl << std::endl;
    }
  }
  OpenCL::clqueue=cl::CommandQueue(OpenCL::clcontext, default_device);
}
