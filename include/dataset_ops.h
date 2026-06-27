#ifndef DATASET_OPS_H
#define DATASET_OPS_H

#include <stddef.h>

float *load_dataset_binary_f32(const char *file_path, size_t expected_float_values);
void one_hot_encode(float *output, const float *labels, int sample_count, int class_count);
void shuffle_indices(int *indices, int count);

#endif // !DATASET_OPS_H

