// Created by AG on 20-03-2026

#ifndef MATH_OPS_H
#define MATH_OPS_H

#include <stddef.h>

void compute_math_add(float *out, const float *term_a, const float *term_b, int total);
void compute_math_subtract(float *out, const float *term_a, const float *term_b, int total);
void compute_math_scale(float *data, float scalar_term, int total);
void compute_math_fill(float *data, float value, int total);
void compute_math_clear(float *data, int total);
void compute_math_copy(float *dest, const float *src, int total);
void compute_math_accumulate(float *dest, const float *src, int total);

void compute_math_matrix_multiplication_nn(float *out, const float *term_a, const float *term_b, int m, int n, int k, int zero_output);
void compute_math_matrix_multiplication_nt(float *out, const float *term_a, const float *term_b, int m, int n, int k, int zero_output);
void compute_math_matrix_multiplication_tn(float *out, const float *term_a, const float *term_b, int m, int n, int k, int zero_output);
void compute_math_matrix_multiplication_tt(float *out, const float *term_a, const float *term_b, int m, int n, int k, int zero_output);

void compute_relu_forward(float *out, const float *in, int total);
void compute_relu_backward(float *input_gradient, const float *in, const float *upstream_gradient, int total);
void compute_softmax_forward(float *out, const float *in, int total);
void compute_softmax_backward(float *input_gradient, const float *softmax_out, const float *upstream_gradient, int vector_size);
void compute_cross_entropy_forward(float *out, const float *predicted, const float *expected, int total);
void compute_cross_entropy_predicted(float *predicted_gradient, const float *predicted_value, const float *expected_value, const float *upstream_gradient, int total);
void compute_cross_entropy_expected(float *expected_Gradient, const float *predicted_value, const float *upstream_gradient, int total);
void compute_param_update(float *parameter, const float *gradient, float scaled_learning_rate, int total);


#endif
