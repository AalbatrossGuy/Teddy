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


