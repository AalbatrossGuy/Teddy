// Created by AG on 17-08-2026

#ifndef GPU_COMPUTE_H
#define GPU_COMPUTE_H

#ifdef SYSTEM_HAS_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stddef.h>

typedef enum {
  OPENCL_KERNEL_MATH_ADD,
  OPENCL_KERNEL_MATH_SUBTRACT,
  OPENCL_KERNEL_MATH_SCALE,
  OPENCL_KERNEL_MATH_FILL,
  OPENCL_KERNEL_MATH_CLEAR,
  OPENCL_KERNEL_MATH_COPY,
  OPENCL_KERNEL_MATH_ACCUMULATE,
  OPENCL_KERNEL_ADD_BIAS,
  OPENCL_KERNEL_ADD_BIAS_GRADIENT,
  OPENCL_KERNEL_MATRIX_MULTIPLICATION_NN,
  OPENCL_KERNEL_MATRIX_MULTIPLICATION_NT,
  OPENCL_KERNEL_MATRIX_MULTIPLICATION_TN,
  OPENCL_KERNEL_MATRIX_MULTIPLICATION_TT,
  OPENCL_KERNEL_RELU_FORWARD,
  OPENCL_KERNEL_RELU_BACKWARD,
  OPENCL_KERNEL_SOFTMAX_FORWARD,
  OPENCL_KERNEL_SOFTMAX_BACKWARD,
  OPENCL_KERNEL_CROSS_ENTROPY_FORWARD,
  OPENCL_KERNEL_CROSS_ENTROPY_PREDICTED,
  OPENCL_KERNEL_CROSS_ENTROPY_EXPECTED,
  OPENCL_KERNEL_PARAM_UPDATE,
  OPENCL_KERNEL_COUNT
} OpenCLKernelId;

typedef struct {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernels[OPENCL_KERNEL_COUNT];
  int initialized;
} OpenCLDevice;

OpenCLDevice *opencl_device_create(const char *kernel_source_path);
void opencl_device_destroy(OpenCLDevice *device);

cl_mem opencl_device_alloc(OpenCLDevice *device, size_t byte_count);
void opencl_device_free(cl_mem buffer);
void opencl_device_upload(OpenCLDevice *device, cl_mem buffer, const void *host_ptr, size_t byte_count);
void opencl_device_download(OpenCLDevice *device, cl_mem buffer, void *host_ptr, size_t byte_count);
void opencl_device_finish(OpenCLDevice *device);

void opencl_device_dispatch_1d(OpenCLDevice *device, OpenCLKernelId kernel_id, size_t global_size);
cl_kernel opencl_device_get_kernel(OpenCLDevice *device, OpenCLKernelId kernel_id);

#endif

#endif
