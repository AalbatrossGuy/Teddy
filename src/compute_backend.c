#include "compute_backend.h"
#include <stdio.h>
#include <stdlib.h>

static ComputeBackend *global_backend = NULL;


ComputeBackend *compute_backend_init(const char *kernel_path) {
  ComputeBackend *backend = (ComputeBackend *)calloc(1, sizeof(ComputeBackend));
  backend->type = COMPUTE_BACKEND_CPU;
  backend->device_handle = NULL;
  printf("[teddy] compiling with CPU...\n");
  return backend;
}

void compute_backend_destroy(ComputeBackend *backend) {
  if (!backend) return;

  if (global_backend == backend) {
    global_backend == NULL;
  }

  free(backend);
}

void compute_backend_finish(ComputeBackend *backend) {
  if (!backend) return;
}

ComputeBackend *compute_backend_global(void) {
  return global_backend;
}

void compute_backend_set_global(ComputeBackend *backend) {
  global_backend = backend;
}
