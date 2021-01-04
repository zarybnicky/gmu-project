cmake_minimum_required (VERSION 3.7)
project(gmu-nn)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gmu-nn
    data.cpp
    convnn.cpp
    mnist.cpp
    cifar10.cpp
    opencl.cpp
    thesis.cpp
)
target_link_libraries(gmu-nn OpenCL::OpenCL ${OpenCV_LIBS})
