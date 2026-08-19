// Created by AG on 17-08-2026

__kernel void kernel_math_add(__global float *out,
                               __global const float *term_a,
                               __global const float *term_b,
                               const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    out[idx] = term_a[idx] + term_b[idx];
  }
}

__kernel void kernel_math_subtract(__global float *out,
                                    __global const float *term_a,
                                    __global const float *term_b,
                                    const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    out[idx] = term_a[idx] - term_b[idx];
  }
}

__kernel void kernel_math_scale(__global float *data,
                                 const float scalar_term,
                                 const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    data[idx] *= scalar_term;
  }
}

__kernel void kernel_math_fill(__global float *data,
                                const float value,
                                const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    data[idx] = value;
  }
}

__kernel void kernel_math_clear(__global float *data,
                                 const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    data[idx] = 0.0f;
  }
}

__kernel void kernel_math_copy(__global float *dest,
                                __global const float *src,
                                const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    dest[idx] = src[idx];
  }
}

__kernel void kernel_math_accumulate(__global float *dest,
                                      __global const float *src,
                                      const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    dest[idx] += src[idx];
  }
}

__kernel void kernel_matrix_multiplication_nn(__global float *out,
                                               __global const float *term_a,
                                               __global const float *term_b,
                                               const int m, const int n, const int k,
                                               const int zero_output) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  if (row < m && column < n) {
    float accumulator = 0.0f;
    for (int inner_row = 0; inner_row < k; inner_row++) {
      accumulator += term_a[row * k + inner_row] * term_b[inner_row * n + column];
    }
    if (zero_output) {
      out[row * n + column] = accumulator;
    } else {
      out[row * n + column] += accumulator;
    }
  }
}

__kernel void kernel_matrix_multiplication_nt(__global float *out,
                                               __global const float *term_a,
                                               __global const float *term_b,
                                               const int m, const int n, const int k,
                                               const int zero_output) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  if (row < m && column < n) {
    float accumulator = 0.0f;
    for (int inner_row = 0; inner_row < k; inner_row++) {
      accumulator += term_a[row * k + inner_row] * term_b[column * k + inner_row];
    }
    if (zero_output) {
      out[row * n + column] = accumulator;
    } else {
      out[row * n + column] += accumulator;
    }
  }
}

__kernel void kernel_matrix_multiplication_tn(__global float *out,
                                               __global const float *term_a,
                                               __global const float *term_b,
                                               const int m, const int n, const int k,
                                               const int zero_output) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  if (row < m && column < n) {
    float accumulator = 0.0f;
    for (int inner_row = 0; inner_row < k; inner_row++) {
      accumulator += term_a[inner_row * m + row] * term_b[inner_row * n + column];
    }
    if (zero_output) {
      out[row * n + column] = accumulator;
    } else {
      out[row * n + column] += accumulator;
    }
  }
}

__kernel void kernel_matrix_multiplication_tt(__global float *out,
                                               __global const float *term_a,
                                               __global const float *term_b,
                                               const int m, const int n, const int k,
                                               const int zero_output) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  if (row < m && column < n) {
    float accumulator = 0.0f;
    for (int inner_row = 0; inner_row < k; inner_row++) {
      accumulator += term_a[inner_row * m + row] * term_b[column * k + inner_row];
    }
    if (zero_output) {
      out[row * n + column] = accumulator;
    } else {
      out[row * n + column] += accumulator;
    }
  }
}

__kernel void kernel_relu_forward(__global float *out,
                                   __global const float *in,
                                   const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    out[idx] = fmax(0.0f, in[idx]);
  }
}

__kernel void kernel_relu_backward(__global float *input_gradient,
                                    __global const float *in,
                                    __global const float *upstream_gradient,
                                    const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    input_gradient[idx] += (in[idx] > 0.0f) ? upstream_gradient[idx] : 0.0f;
  }
}

__kernel void kernel_softmax_forward(__global float *out,
                                      __global const float *in,
                                      const int rows, const int columns) {
  int column = get_global_id(0);
  if (column >= columns) return;

  float max_value = in[column];
  for (int row = 1; row < rows; row++) {
    max_value = fmax(max_value, in[row * columns + column]);
  }

  float exponential_sum = 0.0f;
  for (int row = 0; row < rows; row++) {
    float e = exp(in[row * columns + column] - max_value);
    out[row * columns + column] = e;
    exponential_sum += e;
  }

  float inverse_sum = 1.0f / exponential_sum;
  for (int row = 0; row < rows; row++) {
    out[row * columns + column] *= inverse_sum;
  }
}

__kernel void kernel_softmax_backward(__global float *input_gradient,
                                       __global const float *softmax_out,
                                       __global const float *upstream_gradient,
                                       const int rows, const int columns) {
  int column = get_global_id(0);
  if (column >= columns) return;

  for (int i = 0; i < rows; i++) {
    float partial_sum = 0.0f;
    float si = softmax_out[i * columns + column];
    for (int j = 0; j < rows; j++) {
      float sj = softmax_out[j * columns + column];
      float jacobian_elem = (i == j) ? si * (1.0f - si) : -si * sj;
      partial_sum += jacobian_elem * upstream_gradient[j * columns + column];
    }
    input_gradient[i * columns + column] += partial_sum;
  }
}

__kernel void kernel_add_bias(__global float *out,
                               __global const float *value,
                               __global const float *bias,
                               const int rows, const int columns) {
  int idx = get_global_id(0);
  int total = rows * columns;
  if (idx < total) {
    int row = idx / columns;
    out[idx] = value[idx] + bias[row];
  }
}

__kernel void kernel_add_bias_gradient(__global float *bias_gradient,
                                        __global const float *upstream_gradient,
                                        const int rows, const int columns) {
  int row = get_global_id(0);
  if (row < rows) {
    float sum = 0.0f;
    for (int column = 0; column < columns; column++) {
      sum += upstream_gradient[row * columns + column];
    }
    bias_gradient[row] += sum;
  }
}

__kernel void kernel_cross_entropy_forward(__global float *out,
                                            __global const float *predicted,
                                            __global const float *expected,
                                            const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    if (expected[idx] == 0.0f) {
      out[idx] = 0.0f;
    } else {
      out[idx] = -expected[idx] * log(fmax(predicted[idx], 1e-7f));
    }
  }
}

__kernel void kernel_cross_entropy_predicted(__global float *predicted_gradient,
                                              __global const float *predicted_value,
                                              __global const float *expected_value,
                                              __global const float *upstream_gradient,
                                              const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    float clamped_data = fmax(predicted_value[idx], 1e-7f);
    predicted_gradient[idx] += (-expected_value[idx] / clamped_data) * upstream_gradient[idx];
  }
}

__kernel void kernel_cross_entropy_expected(__global float *expected_gradient,
                                             __global const float *predicted_value,
                                             __global const float *upstream_gradient,
                                             const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    float clamped_data = fmax(predicted_value[idx], 1e-7f);
    expected_gradient[idx] += (-log(clamped_data)) * upstream_gradient[idx];
  }
}

__kernel void kernel_param_update(__global float *parameter,
                                   __global const float *gradient,
                                   const float scaled_learning_rate,
                                   const int total) {
  int idx = get_global_id(0);
  if (idx < total) {
    parameter[idx] -= scaled_learning_rate * gradient[idx];
  }
}
