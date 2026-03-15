#ifndef COMPUTE_BACKEND_H
#define COMPUTE_BACKEND_H

typedef enum {
  COMPUTE_BACKEND_CPU,
  COMPUTE_BACKEND_GPU
} ComputeBackendType;

typedef struct {
  ComputeBackendType type;
  void *device_handle;
} ComputeBackend;

ComputeBackend *compute_backend_init(const char *kernel_path);
void compute_backend_destroy(ComputeBackend *backend);
void compute_backend_finish(ComputeBackend *backend);

ComputeBackend *compute_backend_global(void);
void compute_backend_set_global(ComputeBackend *backend);

#endif
