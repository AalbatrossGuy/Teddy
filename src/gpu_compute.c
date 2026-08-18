// Created by AG on 17-08-2026

#include "gpu_compute.h"

#ifdef SYSTEM_HAS_OPENCL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kernel_names[OPENCL_KERNEL_COUNT] = {
  "kernel_math_add",
  "kernel_math_subtract",
  "kernel_math_scale",
  "kernel_math_fill",
  "kernel_math_clear",
  "kernel_math_copy",
  "kernel_math_accumulate",
  "kernel_matrix_multiplication_nn",
  "kernel_matrix_multiplication_nt",
  "kernel_matrix_multiplication_tn",
  "kernel_matrix_multiplication_tt",
  "kernel_relu_forward",
  "kernel_relu_backward",
  "kernel_softmax_forward",
  "kernel_softmax_backward",
  "kernel_cross_entropy_forward",
  "kernel_cross_entropy_predicted",
  "kernel_cross_entropy_expected",
  "kernel_param_update"
};

static char *read_entire_file(const char *filepath) {
  FILE *file = fopen(filepath, "rb");
  if (!file) {
    fprintf(stderr, "Teddy: failed to open kernel file: %s\n", filepath);
    return NULL;
  }

  fseek(file, 0, SEEK_END);
  long file_length = ftell(file);
  fseek(file, 0, SEEK_SET);

  char *source_buffer = (char *)malloc(file_length + 1);
  fread(source_buffer, 1, file_length, file);
  source_buffer[file_length] = '\0';
  fclose(file);

  return source_buffer;
}

static cl_device_id select_gpu_device(cl_platform_id *out_platform) {
  cl_uint platform_count = 0;
  clGetPlatformIDs(0, NULL, &platform_count);

  if (platform_count == 0) {
    fprintf(stderr, "Teddy: no OpenCL platforms found\n");
    return NULL;
  }

  cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * platform_count);
  clGetPlatformIDs(platform_count, platforms, NULL);

  for (cl_uint p = 0; p < platform_count; p++) {
    cl_uint device_count = 0;
    clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &device_count);

    if (device_count > 0) {
      cl_device_id device;
      clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

      char device_name[256];
      clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);

      char platform_name[256];
      clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);

      printf("Teddy: selected GPU: %s (%s)\n", device_name, platform_name);

      *out_platform = platforms[p];
      free(platforms);
      return device;
    }
  }

  free(platforms);
  fprintf(stderr, "Teddy: no GPU devices found on any platform\n");
  return NULL;
}

OpenCLDevice *opencl_device_create(const char *kernel_source_path) {
  OpenCLDevice *device = (OpenCLDevice *)calloc(1, sizeof(OpenCLDevice));

  device->device = select_gpu_device(&device->platform);
  if (!device->device) {
    free(device);
    return NULL;
  }

  cl_int status;
  device->context = clCreateContext(NULL, 1, &device->device, NULL, NULL, &status);
  if (status != CL_SUCCESS) {
    fprintf(stderr, "Teddy: failed to create OpenCL context: %d\n", status);
    free(device);
    return NULL;
  }

#ifdef CL_VERSION_2_0
  device->queue = clCreateCommandQueueWithProperties(device->context, device->device, NULL, &status);
#else
  device->queue = clCreateCommandQueue(device->context, device->device, 0, &status);
#endif

  if (status != CL_SUCCESS) {
    fprintf(stderr, "Teddy: failed to create command queue: %d\n", status);
    clReleaseContext(device->context);
    free(device);
    return NULL;
  }

  char *kernel_source = read_entire_file(kernel_source_path);
  if (!kernel_source) {
    clReleaseCommandQueue(device->queue);
    clReleaseContext(device->context);
    free(device);
    return NULL;
  }

  const char *source_strings[] = { kernel_source };
  size_t source_lengths[] = { strlen(kernel_source) };
  device->program = clCreateProgramWithSource(device->context, 1, source_strings, source_lengths, &status);
  free(kernel_source);

  if (status != CL_SUCCESS) {
    fprintf(stderr, "Teddy: failed to create program: %d\n", status);
    clReleaseCommandQueue(device->queue);
    clReleaseContext(device->context);
    free(device);
    return NULL;
  }

  status = clBuildProgram(device->program, 1, &device->device, "-cl-fast-relaxed-math", NULL, NULL);
  if (status != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(device->program, device->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *build_log = (char *)malloc(log_size + 1);
    clGetProgramBuildInfo(device->program, device->device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    fprintf(stderr, "Teddy: kernel build failed:\n%s\n", build_log);
    free(build_log);
    clReleaseProgram(device->program);
    clReleaseCommandQueue(device->queue);
    clReleaseContext(device->context);
    free(device);
    return NULL;
  }

  for (int i = 0; i < OPENCL_KERNEL_COUNT; i++) {
    device->kernels[i] = clCreateKernel(device->program, kernel_names[i], &status);
    if (status != CL_SUCCESS) {
      fprintf(stderr, "Teddy: failed to create kernel '%s': %d\n", kernel_names[i], status);
      for (int j = 0; j < i; j++) {
        clReleaseKernel(device->kernels[j]);
      }
      clReleaseProgram(device->program);
      clReleaseCommandQueue(device->queue);
      clReleaseContext(device->context);
      free(device);
      return NULL;
    }
  }

  device->initialized = 1;
  printf("Teddy: GPU backend initialized with %d kernels\n", OPENCL_KERNEL_COUNT);
  return device;
}

void opencl_device_destroy(OpenCLDevice *device) {
  if (!device) {
    return;
  }

  for (int i = 0; i < OPENCL_KERNEL_COUNT; i++) {
    if (device->kernels[i]) {
      clReleaseKernel(device->kernels[i]);
    }
  }

  if (device->program) {
    clReleaseProgram(device->program);
  }

  if (device->queue) {
    clReleaseCommandQueue(device->queue);
  }

  if (device->context) {
    clReleaseContext(device->context);
  }

  free(device);
}

cl_mem opencl_device_alloc(OpenCLDevice *device, size_t byte_count) {
  cl_int status;
  cl_mem buffer = clCreateBuffer(device->context, CL_MEM_READ_WRITE, byte_count, NULL, &status);
  if (status != CL_SUCCESS) {
    fprintf(stderr, "Teddy: GPU allocation failed for %zu bytes: %d\n", byte_count, status);
    return NULL;
  }
  return buffer;
}

void opencl_device_free(cl_mem buffer) {
  if (buffer) {
    clReleaseMemObject(buffer);
  }
}

void opencl_device_upload(OpenCLDevice *device, cl_mem buffer, const void *host_ptr, size_t byte_count) {
  clEnqueueWriteBuffer(device->queue, buffer, CL_TRUE, 0, byte_count, host_ptr, 0, NULL, NULL);
}

void opencl_device_download(OpenCLDevice *device, cl_mem buffer, void *host_ptr, size_t byte_count) {
  clEnqueueReadBuffer(device->queue, buffer, CL_TRUE, 0, byte_count, host_ptr, 0, NULL, NULL);
}

void opencl_device_finish(OpenCLDevice *device) {
  clFinish(device->queue);
}

void opencl_device_dispatch_1d(OpenCLDevice *device, OpenCLKernelId kernel_id, size_t global_size) {
  size_t local_size = 256;
  size_t aligned_global = ((global_size + local_size - 1) / local_size) * local_size;
  clEnqueueNDRangeKernel(device->queue, device->kernels[kernel_id], 1, NULL,
                         &aligned_global, &local_size, 0, NULL, NULL);
}

cl_kernel opencl_device_get_kernel(OpenCLDevice *device, OpenCLKernelId kernel_id) {
  return device->kernels[kernel_id];
}

#endif
