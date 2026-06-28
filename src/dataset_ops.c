#include "dataset_ops.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


float *load_dataset_binary_f32(const char *file_path, size_t expected_float_values){
  FILE *dataset_file = fopen(file_path, "rb");

  if (!dataset_file){
    fprintf(stderr, "Teddy: failed to open dataset file: %s\n", file_path);
    return NULL;
  }

  fseek(dataset_file, 0, SEEK_END);
  long dataset_file_size = ftell(dataset_file);
  fseek(dataset_file, 0, SEEK_SET);

  size_t expected_bytes_to_read = expected_float_values * sizeof(float);
  size_t bytes_to_read = (size_t) dataset_file_size < expected_bytes_to_read ? (size_t) dataset_file_size : expected_bytes_to_read;

  float *file_data = (float *) malloc(expected_bytes_to_read);
  memset(file_data, 0, expected_bytes_to_read);
  size_t bytes_read = fread(file_data, 1, bytes_to_read, dataset_file);
  fclose(dataset_file);

  printf("Teddy: file %s (%zu floats, %zu bytes) loaded!\n", file_path, expected_float_values, bytes_read);
  return file_data;
}

void one_hot_encode(float *output, const float *labels, int sample_count, int class_count){
  memset(output, 0, sizeof(float) * sample_count * class_count);
  for (int i = 0; i < sample_count; i++) {
      int label_index = (int)labels[i];
      if (label_index >= 0 && label_index < class_count){
          output[i * class_count + label_index] = 1.0f;
      }
  }
}


void dataset_shuffle_indices(int *indices, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}
