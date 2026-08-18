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
                                               const int m, const int n, const int k) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  if (row < m && column < n) {
    float accumulator = 0.0f;
    for (int inner_row = 0; inner_row < k; inner_row++) {
      accumulator += term_a[row * k + inner_row] * term_b[inner_row * n + column];
    }
    out[row * n + column] += accumulator;
  }
}

__kernel void kernel_matrix_multiplication_nt(__global float *out,
                                               __global const float *term_a,
                                               __global const float *term_b,
                                               const int m, const int n, const int k) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  if (row < m && column < n) {
    float accumulator = 0.0f;
    for (int inner_row = 0; inner_row < k; inner_row++) {
      accumulator += term_a[row * k + inner_row] * term_b[column * k + inner_row];
    }
    out[row * n + column] += accumulator;
  }
}

__kernel void kernel_matrix_multiplication_tn(__global float *out,
                                               __global const float *term_a,
                                               __global const float *term_b,
                                               const int m, const int n, const int k) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  if (row < m && column < n) {
    float accumulator = 0.0f;
    for (int inner_row = 0; inner_row < k; inner_row++) {
      accumulator += term_a[inner_row * m + row] * term_b[inner_row * n + column];
    }
    out[row * n + column] += accumulator;
  }
}

__kernel void kernel_matrix_multiplication_tt(__global float *out,
                                               __global const float *term_a,
                                               __global const float *term_b,
                                               const int m, const int n, const int k) {
  int row = get_global_id(0);
  int column = get_global_id(1);
  if (row < m && column < n) {
    float accumulator = 0.0f;
    for (int inner_row = 0; inner_row < k; inner_row++) {
      accumulator += term_a[inner_row * m + row] * term_b[column * k + inner_row];
    }
    out[row * n + column] += accumulator;
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
                                      const int total) {
  float max_value = -INFINITY;
  for (int i = 0; i < total; i++) {
    max_value = fmax(max_value, in[i]);
  }

  float exponential_sum = 0.0f;
  for (int i = 0; i < total; i++) {
    out[i] = exp(in[i] - max_value);
    exponential_sum += out[i];
  }

  float inverse_sum = 1.0f / exponential_sum;
  for (int i = 0; i < total; i++) {
    out[i] *= inverse_sum;
  }
}

__kernel void kernel_softmax_backward(__global float *input_gradient,
                                       __global const float *softmax_out,
                                       __global const float *upstream_gradient,
                                       const int vector_size) {
  for (int i = 0; i < vector_size; i++) {
    float partial_sum = 0.0f;
    for (int j = 0; j < vector_size; j++) {
      float jacobian_elem;
      if (i == j) {
        jacobian_elem = softmax_out[i] * (1.0f - softmax_out[i]);
      } else {
        jacobian_elem = -softmax_out[i] * softmax_out[j];
      }
      partial_sum += jacobian_elem * upstream_gradient[j];
    }
    input_gradient[i] += partial_sum;
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
