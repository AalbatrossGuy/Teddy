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

