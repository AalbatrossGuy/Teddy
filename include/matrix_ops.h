// Created by AG on 15-03-2026

#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include "compute_backend.h"
#include <stddef.h>

typedef struct {
  int rows;
  int columns;
  float *host_data;
} Matrix;

Matrix *matrix_create(int rows, int columns);
void matrix_destroy(Matrix *mat);
Matrix *matrix_clone(const Matrix *source);
void matrix_copy(Matrix *dest, const Matrix *src);

void matrix_download(const Matrix *mat, float *data);
void matrix_upload(Matrix *mat, const float *data);

void matrix_clear(Matrix *mat);
void matrix_fill(Matrix *mat, float value);
void matrix_fill_random(Matrix *mat, float lower, float upper);

void matrix_add(Matrix *out, const Matrix *mat_a, const Matrix *mat_b);
void matrix_sub(Matrix *out, const Matrix *mat_a, const Matrix *mat_b);
void matrix_accumulate(Matrix *dest, const Matrix *src);
void matrix_scale(Matrix *mat, float scalar);
void matrix_multiply(Matrix *out, const Matrix *mat_a, const Matrix *mat_b, int transpose_a, int transpose_b, int zero_output);

void matrix_reLU(Matrix *out, const Matrix *in);
void matrix_softmax(Matrix *out, const Matrix *in);
void matrix_cross_entropy(Matrix *out, const Matrix *predicted, const Matrix *expected);
void matrix_reLU_gradient(Matrix *input_grad, const Matrix *input_val, const Matrix *upstream_grad);
void matrix_softmax_gradient(Matrix *input_grad, const Matrix *softmax_out, const Matrix *upstream_grad);
void matrix_cross_entropy_gradient_predicted(Matrix *predicted_grad, const Matrix *predicted_val, const Matrix *expected_val, const Matrix *upstream_grad);
void matrix_cross_entropy_gradient_expected(Matrix *expected_grad, const Matrix *predicted_val, const Matrix *upstream_grad);

void matrix_param_update(Matrix *parameter, const Matrix *gradient, float scaled_learning_rate);
float matrix_sum(const Matrix *mat);
int matrix_argmax(const Matrix *mat);

static inline int matrix_element_count(const Matrix *mat) {
  return mat->rows * mat->columns;
}

#endif
