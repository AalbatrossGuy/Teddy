#include "math_ops.h"
#include <stddef.h>
#include <string.h>
#include <math.h>

void compute_math_add(float *out, const float *term_a, const float *term_b, int total) {
  for (int i = 0; i < total; i++) {
    out[i] = term_a[i] + term_b[i];
  }
}

void compute_math_subtract(float *out, const float *term_a, const float *term_b, int total) {
  for (int i = 0; i < total; i++){
       out[i] = term_a[i] - term_b[i];
  }
}

void compute_math_scale(float *data, float scalar_term, int total) {
  for (int i = 0; i < total; i++) {
    data[i]  *= scalar_term;
  }
}

void compute_math_fill(float *data, float value, int total) {
  for (int i = 0; i < total; i++) {
    data[i] = value;
  }
}

void compute_math_clear(float *data, int total) {
  memset(data, 0, sizeof(float) * total);
}

void compute_math_copy(float *dest, const float *src, int total) {
  memcpy(dest, src, sizeof(float) * total);
}

void compute_math_accumulate(float *dest, const float *src, int total) {
  for (int i = 0; i < total; i++) {
    dest[i] += src[i];
  }
}

void compute_math_matrix_multiplication_nn(float *out, const float *term_a, const float *term_b, int m, int n, int k, int zero_output) {
  if (zero_output) {
    memset(out, 0, sizeof(float) * m * n);
  }

  for (int row = 0; row < m; row++) {
    for(int inner_row = 0; inner_row < k; inner_row++) {
      for (int column = 0; column < n; column++) {
        out[row * n + column] += term_a[row * k + inner_row] * term_b[inner_row * n + column];
      }
    }
  }
}


void compute_math_matrix_multiplication_nt(float *out, const float *term_a, const float *term_b, int m, int n, int k, int zero_output) {
  if (zero_output) {
    memset(out, 0, sizeof(float) * m * n);
  }

  for (int row = 0; row < m; row++) {
    for(int inner_row = 0; inner_row < k; inner_row++) {
      for (int column = 0; column < n; column++) {
        out[row * n + column] += term_a[row * k + inner_row] * term_b[column * k + inner_row];
      }
    }
  }
}


void compute_math_matrix_multiplication_tn(float *out, const float *term_a, const float *term_b, int m, int n, int k, int zero_output) {
  if (zero_output) {
    memset(out, 0, sizeof(float) * m * n);
  }

  for (int inner_row = 0; inner_row < k; inner_row++) {
    for(int row = 0; row < m; row++) {
      for (int column = 0; column < n; column++) {
        out[row * n + column] += term_a[inner_row * m + row] * term_b[inner_row * n + column];
      }
    }
  }
}


void compute_math_matrix_multiplication_tt(float *out, const float *term_a, const float *term_b, int m, int n, int k, int zero_output) {
  if (zero_output) {
    memset(out, 0, sizeof(float) * m * n);
  }

  for (int row = 0; row < m; row++) {
    for(int column = 0; column < n; column++) {
      for (int inner_row = 0; inner_row < k; inner_row++) {
        out[row * n + column] += term_a[inner_row * m + row] * term_b[column * k + inner_row];
      }
    }
  }
}

void compute_relu_forward(float *out, const float *in, int total) {
  for (int i = 0; i < total; i++) {
    out[i] = in[i] > 0.0f ? in[i]: 0.0f;
  }
}

void compute_relu_backward(float *input_gradient, const float *in, const float *upstream_gradient, int total) {
  for (int i = 0; i < total; i++) {
    input_gradient[i] += (in[i] > 0.0f) ? upstream_gradient[i] : 0.0f;
  }
}

void compute_softmax_forward(float *out, const float *in, int total) {
  float max_value = in[0];

  for (int i = 1; i < total; i++) {
    if (in[i] > max_value) {
      max_value = in[i];
    }
  }

  float exponential_sum = 0.0f;
  for (int i = 0; i < total; i++) {
    out[i] = expf(in[i] - max_value);
    exponential_sum += out[i];
  }

  float inverse_sum = 1.0f / exponential_sum;
  for (int i = 0; i < total; i++) {
    out[i] *= inverse_sum;
  }
}

void compute_softmax_backward(float *input_gradient, const float *softmax_out, const float *upstream_gradient, int vector_size) {
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

void compute_cross_entropy_forward(float *out, const float *predicted, const float *expected, int total) {
  for (int i = 0; i < total; i++) {
    if (expected[i] == 0.0f) {
      out[i] = 0.0f;
    } else {
      float clamped_data = predicted[i] > 1e-7f ? predicted[i] : 1e-7f;
      out[i] = -expected[i] * logf(clamped_data);
    }
  }
}

void compute_cross_entropy_predicted(float *predicted_gradient, const float *predicted_value, const float *expected_value, const float *upstream_gradient, int total) {
  for (int i = 0; i < total; i++) {
    float clamped_data = predicted_value[i] > 1e-7f ? predicted_value[i] : 1e-7f;
    predicted_gradient[i] += (-expected_value[i] / clamped_data) * upstream_gradient[i];
  }
}

void compute_cross_entropy_expected(float *expected_Gradient, const float *predicted_value, const float *upstream_gradient, int total) {
  for (int i = 0; i < total; i++) {
    float clamped_data = predicted_value[i] > 1e-7f ? predicted_value[i] : 1e-7f;
    expected_Gradient[i] += (-logf(clamped_data)) * upstream_gradient[i];
  }
}

void compute_param_update(float *parameter, const float *gradient, float scaled_learning_rate, int total) {
  for (int i = 0; i < total; i++) {
    parameter[i] -= scaled_learning_rate * gradient[i];
  }
}
