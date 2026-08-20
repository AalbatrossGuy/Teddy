#include "compute_backend.h"
#ifdef SYSTEM_HAS_OPENCL
#include "gpu_compute.h"
#endif
#include <stdio.h>
#include <stdlib.h>

static ComputeBackend *global_backend = NULL;


ComputeBackend *compute_backend_init(const char *kernel_path) {
  ComputeBackend *backend = (ComputeBackend *)calloc(1, sizeof(ComputeBackend));

#ifdef SYSTEM_HAS_OPENCL
  OpenCLDevice *gpu = opencl_device_create(kernel_path);
  if (gpu) {
    backend->type = COMPUTE_BACKEND_GPU;
    backend->device_handle = gpu;
    return backend;
  }
  printf("Teddy: GPU unavailable, falling back to CPU\n");
#else
  (void)kernel_path;
  printf("Teddy: Compiling with CPU...\n");
#endif

  backend->type = COMPUTE_BACKEND_CPU;
  backend->device_handle = NULL;
  return backend;
}

void compute_backend_destroy(ComputeBackend *backend) {
  if (!backend) return;

#ifdef SYSTEM_HAS_OPENCL
  if (backend->type == COMPUTE_BACKEND_GPU && backend->device_handle) {
    opencl_device_destroy((OpenCLDevice *)backend->device_handle);
  }
#endif

  if (global_backend == backend) {
    global_backend = NULL;
  }

  free(backend);
}

void compute_backend_finish(ComputeBackend *backend) {
  if (!backend) return;

#ifdef SYSTEM_HAS_OPENCL
  if (backend->type == COMPUTE_BACKEND_GPU && backend->device_handle) {
    opencl_device_finish((OpenCLDevice *)backend->device_handle);
  }
#endif
}

ComputeBackend *compute_backend_global(void) {
  return global_backend;
}

void compute_backend_set_global(ComputeBackend *backend) {
  global_backend = backend;
}
