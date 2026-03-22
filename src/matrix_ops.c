// Created by AG on 22-03-2026

#include "matrix_ops.h"
#include "math_ops.h"
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>


static int seeded = 0;

Matrix *matrix_create(int rows, int columns) {
  Matrix *matrix = (Matrix *) calloc(1, sizeof(Matrix));

  matrix->rows = rows;
  matrix->columns = columns;
  int total = rows * columns;

  matrix->host_data = (float *) calloc(total, sizeof(float));
  return matrix;
}

void matrix_destroy(Matrix *mat) {
  if (!mat) {
    return;
  }

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
  memcpy(mat->host_data, data, sizeof(float) * mat->rows * mat->columns);
}

void matrix_download(const Matrix *mat, float *data) {
  memcpy(data, mat->host_data, sizeof(float) * mat->rows * mat->columns);
}

void matrix_clear(Matrix *mat) {
  int total = mat->rows * mat->columns;
  compute_math_clear(mat->host_data, total);
}

void matrix_fill(Matrix *mat, float value) {
  int total = mat->rows * mat->columns;
  compute_math_fill(mat->host_data, value, total);
}

void matrix_copy(Matrix *dest, const Matrix *src) {
  int total = src->rows * src->columns;
  compute_math_copy(dest->host_data, src->host_data, total);
}

void matrix_add(Matrix *out, const Matrix *mat_a, const Matrix *mat_b) {
  int total = out->rows * out->columns;
  compute_math_add(out->host_data, mat_a->host_data, mat_b->host_data, total);
}

void matrix_sub(Matrix *out, const Matrix *mat_a, const Matrix *mat_b) {
  int total = out->rows * out->columns;
  compute_math_subtract(out->host_data, mat_a->host_data, mat_b->host_data, total);
}

void matrix_accumulate(Matrix *dest, const Matrix *src) {
  int total = dest->rows * dest->columns;
  compute_math_accumulate(dest->host_data, src->host_data, total);
}

void matrix_scale(Matrix *mat, float scalar) {
  int total = mat->rows * mat->columns;
  compute_math_scale(mat->host_data, scalar, total);
}

void matrix_multiply(Matrix *out, const Matrix *mat_a, const Matrix *mat_b, int transpose_a, int transpose_b, int zero_output) {
  int matrix_a_rows = transpose_a ? mat_a->columns : mat_a ->rows;
  int matrix_a_columns = transpose_a ? mat_a->rows : mat_a->columns;
  int matrix_b_columns = transpose_b ? mat_b->rows : mat_b->columns;

  int m = matrix_a_rows;
  int n = matrix_b_columns;
  int k = matrix_a_columns;

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
  compute_relu_forward( out->host_data, in->host_data, total);
}

void matrix_softmax(Matrix *out, const Matrix *in) {
  int total = in->rows * in->columns;
  compute_softmax_forward(out->host_data, in->host_data, total);
}

void matrix_cross_entropy(Matrix *out, const Matrix *predicted, const Matrix *expected) {
  int total = predicted->rows * predicted->columns;
  compute_cross_entropy_forward(out->host_data, predicted->host_data, expected->host_data, total);
}
