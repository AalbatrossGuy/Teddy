// Created by AG on 22-03-2026

#include "matrix_ops.h"
#include "math_ops.h"
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef SYSTEM_HAS_OPENCL
static inline int is_gpu_mode(void) {
  ComputeBackend *backend = compute_backend_global();
  return backend && backend->type == COMPUTE_BACKEND_GPU;
}

static inline OpenCLDevice *get_gpu(void) {
  return (OpenCLDevice *)compute_backend_global()->device_handle;
}
#endif

Matrix *matrix_create(int rows, int columns) {
  Matrix *matrix = (Matrix *) calloc(1, sizeof(Matrix));

  matrix->rows = rows;
  matrix->columns = columns;
  int total = rows * columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    matrix->device_buffer = opencl_device_alloc(gpu, sizeof(float) * total);
    matrix->host_data = NULL;

    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_MATH_CLEAR);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrix->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_MATH_CLEAR, total);
    return matrix;
  }
#endif

  matrix->host_data = (float *) calloc(total, sizeof(float));
  return matrix;
}

void matrix_destroy(Matrix *mat) {
  if (!mat) {
    return;
  }

#ifdef SYSTEM_HAS_OPENCL
  if (mat->device_buffer) {
    opencl_device_free(mat->device_buffer);
  }
#endif

  if (mat->host_data) {
    free(mat->host_data);
  }

  free(mat);
}

Matrix *matrix_clone(const Matrix *source) {
  Matrix *cpy = matrix_create(source->rows, source->columns);
  matrix_copy(cpy, source);
  return cpy;
}

void matrix_upload(Matrix *mat, const float *data) {
#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    opencl_device_upload(get_gpu(), mat->device_buffer, data, sizeof(float) * mat->rows * mat->columns);
    return;
  }
#endif

  memcpy(mat->host_data, data, sizeof(float) * mat->rows * mat->columns);
}

void matrix_download(const Matrix *mat, float *data) {
#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    opencl_device_download(get_gpu(), mat->device_buffer, data, sizeof(float) * mat->rows * mat->columns);
    return;
  }
#endif

  memcpy(data, mat->host_data, sizeof(float) * mat->rows * mat->columns);
}

void matrix_clear(Matrix *mat) {
  int total = mat->rows * mat->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_MATH_CLEAR);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_MATH_CLEAR, total);
    return;
  }
#endif

  compute_math_clear(mat->host_data, total);
}

void matrix_fill(Matrix *mat, float value) {
  int total = mat->rows * mat->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_MATH_FILL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(float), &value);
    clSetKernelArg(kernel, 2, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_MATH_FILL, total);
    return;
  }
#endif

  compute_math_fill(mat->host_data, value, total);
}

void matrix_fill_random(Matrix *mat, float lower, float upper) {
  int total = mat->rows * mat->columns;
  float *data = (float *)malloc(sizeof(float) * total);
  float range = upper - lower;

  for (int i = 0; i < total; i++) {
    data[i] = lower + ((float)rand() / (float)RAND_MAX) * range;
  }
  matrix_upload(mat, data);
  free(data);
}

void matrix_copy(Matrix *dest, const Matrix *src) {
  int total = src->rows * src->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_MATH_COPY);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &src->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_MATH_COPY, total);
    return;
  }
#endif

  compute_math_copy(dest->host_data, src->host_data, total);
}

void matrix_add(Matrix *out, const Matrix *mat_a, const Matrix *mat_b) {
  int total = out->rows * out->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_MATH_ADD);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &mat_a->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &mat_b->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_MATH_ADD, total);
    return;
  }
#endif

  compute_math_add(out->host_data, mat_a->host_data, mat_b->host_data, total);
}

void matrix_sub(Matrix *out, const Matrix *mat_a, const Matrix *mat_b) {
  int total = out->rows * out->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_MATH_SUBTRACT);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &mat_a->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &mat_b->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_MATH_SUBTRACT, total);
    return;
  }
#endif

  compute_math_subtract(out->host_data, mat_a->host_data, mat_b->host_data, total);
}

void matrix_accumulate(Matrix *dest, const Matrix *src) {
  int total = dest->rows * dest->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_MATH_ACCUMULATE);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &src->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_MATH_ACCUMULATE, total);
    return;
  }
#endif

  compute_math_accumulate(dest->host_data, src->host_data, total);
}

void matrix_add_bias(Matrix *out, const Matrix *value, const Matrix *bias) {
#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_ADD_BIAS);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &value->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &out->rows);
    clSetKernelArg(kernel, 4, sizeof(int), &out->columns);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_ADD_BIAS, out->rows * out->columns);
    return;
  }
#endif

  compute_add_bias(out->host_data, value->host_data, bias->host_data, out->rows, out->columns);
}

void matrix_add_bias_gradient(Matrix *bias_gradient, const Matrix *upstream_gradient) {
#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_ADD_BIAS_GRADIENT);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bias_gradient->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &upstream_gradient->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &upstream_gradient->rows);
    clSetKernelArg(kernel, 3, sizeof(int), &upstream_gradient->columns);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_ADD_BIAS_GRADIENT, upstream_gradient->rows);
    return;
  }
#endif

  compute_add_bias_gradient(bias_gradient->host_data, upstream_gradient->host_data, upstream_gradient->rows, upstream_gradient->columns);
}

void matrix_scale(Matrix *mat, float scalar) {
  int total = mat->rows * mat->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_MATH_SCALE);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(float), &scalar);
    clSetKernelArg(kernel, 2, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_MATH_SCALE, total);
    return;
  }
#endif

  compute_math_scale(mat->host_data, scalar, total);
}

void matrix_multiply(Matrix *out, const Matrix *mat_a, const Matrix *mat_b, int transpose_a, int transpose_b, int zero_output) {
  int matrix_a_rows = transpose_a ? mat_a->columns : mat_a ->rows;
  int matrix_a_columns = transpose_a ? mat_a->rows : mat_a->columns;
  int matrix_b_columns = transpose_b ? mat_b->rows : mat_b->columns;

  int m = matrix_a_rows;
  int n = matrix_b_columns;
  int k = matrix_a_columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();

    OpenCLKernelId kernel_id;
    if (!transpose_a && !transpose_b) {
      kernel_id = OPENCL_KERNEL_MATRIX_MULTIPLICATION_NN;
    } else if (!transpose_a && transpose_b) {
      kernel_id = OPENCL_KERNEL_MATRIX_MULTIPLICATION_NT;
    } else if (transpose_a && !transpose_b) {
      kernel_id = OPENCL_KERNEL_MATRIX_MULTIPLICATION_TN;
    } else {
      kernel_id = OPENCL_KERNEL_MATRIX_MULTIPLICATION_TT;
    }

    cl_kernel kernel = opencl_device_get_kernel(gpu, kernel_id);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &mat_a->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &mat_b->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &m);
    clSetKernelArg(kernel, 4, sizeof(int), &n);
    clSetKernelArg(kernel, 5, sizeof(int), &k);
    clSetKernelArg(kernel, 6, sizeof(int), &zero_output);

    size_t global_work_size[2] = { (size_t)m, (size_t)n };
    clEnqueueNDRangeKernel(gpu->queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    return;
  }
#endif

  if (!transpose_a && !transpose_b) {
    compute_math_matrix_multiplication_nn(out->host_data, mat_a->host_data, mat_b->host_data, m, n, k, zero_output);
  } else if (!transpose_a && transpose_b) {
    compute_math_matrix_multiplication_nt(out->host_data, mat_a->host_data, mat_b->host_data, m, n, k, zero_output);
  } else if(transpose_a && !transpose_b) {
    compute_math_matrix_multiplication_tn(out->host_data, mat_a->host_data, mat_b->host_data, m, n, k, zero_output);
  } else {
    compute_math_matrix_multiplication_tt(out->host_data, mat_a->host_data, mat_b->host_data, m, n, k, zero_output);
  }
}

void matrix_reLU(Matrix *out, const Matrix *in) {
  int total = in->rows * in->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_RELU_FORWARD);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &in->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_RELU_FORWARD, total);
    return;
  }
#endif

  compute_relu_forward( out->host_data, in->host_data, total);
}

void matrix_softmax(Matrix *out, const Matrix *in) {
#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_SOFTMAX_FORWARD);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &in->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &in->rows);
    clSetKernelArg(kernel, 3, sizeof(int), &in->columns);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_SOFTMAX_FORWARD, in->columns);
    return;
  }
#endif

  compute_softmax_forward(out->host_data, in->host_data, in->rows, in->columns);
}

void matrix_cross_entropy(Matrix *out, const Matrix *predicted, const Matrix *expected) {
  int total = predicted->rows * predicted->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_CROSS_ENTROPY_FORWARD);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &predicted->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &expected->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_CROSS_ENTROPY_FORWARD, total);
    return;
  }
#endif

  compute_cross_entropy_forward(out->host_data, predicted->host_data, expected->host_data, total);
}

void matrix_reLU_gradient(Matrix *input_grad, const Matrix *input_val, const Matrix *upstream_grad) {
  int total = input_val->rows * input_val->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_RELU_BACKWARD);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_grad->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_val->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &upstream_grad->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_RELU_BACKWARD, total);
    return;
  }
#endif

  compute_relu_backward(input_grad->host_data, input_val->host_data, upstream_grad->host_data, total);
}

void matrix_softmax_gradient(Matrix *input_grad, const Matrix *softmax_out, const Matrix *upstream_grad) {
#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_SOFTMAX_BACKWARD);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_grad->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &softmax_out->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &upstream_grad->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &softmax_out->rows);
    clSetKernelArg(kernel, 4, sizeof(int), &softmax_out->columns);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_SOFTMAX_BACKWARD, softmax_out->columns);
    return;
  }
#endif

  compute_softmax_backward(input_grad->host_data, softmax_out->host_data, upstream_grad->host_data, softmax_out->rows, softmax_out->columns);
}

void matrix_cross_entropy_gradient_predicted(Matrix *predicted_grad, const Matrix *predicted_val, const Matrix *expected_val, const Matrix *upstream_grad) {
  int total = predicted_val->rows * predicted_val->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_CROSS_ENTROPY_PREDICTED);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &predicted_grad->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &predicted_val->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &expected_val->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &upstream_grad->device_buffer);
    clSetKernelArg(kernel, 4, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_CROSS_ENTROPY_PREDICTED, total);
    return;
  }
#endif

  compute_cross_entropy_predicted(predicted_grad->host_data,predicted_val->host_data, expected_val->host_data, upstream_grad->host_data, total);
}

void matrix_cross_entropy_gradient_expected(Matrix *expected_grad, const Matrix *predicted_val, const Matrix *upstream_grad) {
  int total = predicted_val->rows * predicted_val->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_CROSS_ENTROPY_EXPECTED);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &expected_grad->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &predicted_val->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &upstream_grad->device_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_CROSS_ENTROPY_EXPECTED, total);
    return;
  }
#endif

  compute_cross_entropy_expected(expected_grad->host_data, predicted_val->host_data, upstream_grad->host_data, total);
}

void matrix_param_update(Matrix *parameter, const Matrix *gradient, float scaled_learning_rate) {
  int total = parameter->rows * parameter->columns;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    OpenCLDevice *gpu = get_gpu();
    cl_kernel kernel = opencl_device_get_kernel(gpu, OPENCL_KERNEL_PARAM_UPDATE);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &parameter->device_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &gradient->device_buffer);
    clSetKernelArg(kernel, 2, sizeof(float), &scaled_learning_rate);
    clSetKernelArg(kernel, 3, sizeof(int), &total);
    opencl_device_dispatch_1d(gpu, OPENCL_KERNEL_PARAM_UPDATE, total);
    return;
  }
#endif

  compute_param_update(parameter->host_data, gradient->host_data, scaled_learning_rate, total);
}

float matrix_sum(const Matrix *mat) {
  int total = mat->rows * mat->columns;
  float *host_buffer;
  int free_required = 0;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    host_buffer = (float *)malloc(sizeof(float) * total);
    opencl_device_download(get_gpu(), mat->device_buffer, host_buffer, sizeof(float) * total);
    free_required = 1;
  } else
#endif
  {
    host_buffer = mat->host_data;
  }

  float accumlator = 0.0f;

  for (int i = 0; i < total; i++) {
    accumlator += host_buffer[i];
  }

  if (free_required) {
    free(host_buffer);
  }

  return accumlator;
}

int matrix_argmax(const Matrix *mat) {
  int total = mat->rows * mat->columns;
  float *host_buffer;
  int free_required = 0;

#ifdef SYSTEM_HAS_OPENCL
  if (is_gpu_mode()) {
    host_buffer = (float *)malloc(sizeof(float) * total);
    opencl_device_download(get_gpu(), mat->device_buffer, host_buffer, sizeof(float) * total);
    free_required = 1;
  } else
#endif
  {
    host_buffer = mat->host_data;
  }

  int best_index = 0;
  float best_value = host_buffer[0];

  for(int i = 1; i < total; i++) {
    if (host_buffer[i] > best_value) {
      best_value = host_buffer[i];
      best_index = i;
    }
  }

  if (free_required) {
    free(host_buffer);
  }

  return best_index;
}
