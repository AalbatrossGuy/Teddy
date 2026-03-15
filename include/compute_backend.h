#ifndef COMPUTE_BACKEND_H
#define COMPUTE_BACKEND_H

typedef enum {
  MG_BACKEND_CPU,
  MG_BACKEND_GPU
} MgBackendType;

typedef struct {
  MgBackendType type;
  void *device_handle;
} MgBackend;

MgBackend *mg_backend_init(const char *kernel_path);
void mg_backend_destroy(MgBackend *backend);
void mg_backend_finish(MgBackend *backend);

MgBackend *mg_backend_global(void);
void mg_backend_set_global(MgBackend *backend);

#endif
