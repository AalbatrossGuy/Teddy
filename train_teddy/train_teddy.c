// Created by AG on 13-08-2026
#include "compute_backend.h"
#include "matrix_ops.h"
#include "computation_engine.h"
#include "model_train.h"
#include "dataset_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Model Defitions
#define model_dimm 784
#define model_num_classes 10
#define model_training_data_count 60000
#define model_test_data_count 10000
#define model_hidden_layer 16
#define model_batch_size 500
#define model_epochs 10
#define model_learning_rate 0.25f

static void model_draw_digit(const float *pixels) {
  for (int row = 0; row < 28; row++) {
    for (int column = 0; column < 28; column++) {
      int grayscale_image_data = (int) pixels[row * 28 + column] * 23.0f;
      printf("\033[48;5;%dm \033[0m", 232 +  grayscale_image_data);
    }
    printf("\n");
  }
}

static ComputationGraph *model_build(int batch_size) {
  ComputationGraph *graph = computation_graph_create();
  GraphNode *input_node = computation_graph_variable(graph, model_dimm, batch_size, GRAPH_NODE_INPUT);
  GraphNode *initial_weight = computation_graph_variable(graph, model_hidden_layer, model_dimm, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *initial_bias_value = computation_graph_variable(graph, model_hidden_layer, 1, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *initial_preactivation_value = computation_graph_matrix_multiply(graph, initial_weight, input_node, 0);
  GraphNode *biased_value = computation_graph_add_bias(graph, initial_preactivation_value, initial_bias_value, 0);
  GraphNode *activation_value = computation_graph_reLU(graph, biased_value, 0);

  GraphNode *w1 = computation_graph_variable(graph, model_hidden_layer, model_hidden_layer, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *bias1 = computation_graph_variable(graph, model_hidden_layer, 1, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *preactivation_value1 = computation_graph_matrix_multiply(graph, w1, activation_value, 0);
  GraphNode *biased_value_1 = computation_graph_add_bias(graph, preactivation_value1, bias1, 0);
  GraphNode *preresidual1 = computation_graph_reLU(graph, biased_value_1, 0);

  GraphNode *residual_sum_value = computation_graph_add(graph, preresidual1, activation_value, 0);

  GraphNode *w2 = computation_graph_variable(graph, model_num_classes, model_hidden_layer, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *bias2 = computation_graph_variable(graph, model_num_classes, 1, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *preactivation_value2 = computation_graph_matrix_multiply(graph, w2, residual_sum_value, 0);
  GraphNode *biased_value_2 = computation_graph_add_bias(graph, preactivation_value2, bias2, 0);

  GraphNode *output = computation_graph_softmax(graph, biased_value_2, GRAPH_NODE_OUTPUT);
  GraphNode *target_value = computation_graph_variable(graph, model_num_classes, batch_size, GRAPH_NODE_TARGET);
  GraphNode *loss_value = computation_graph_cross_entropy(graph, output, target_value, GRAPH_NODE_LOSS);


  (void) loss_value;
  model_weight_matrix(initial_weight);
  model_weight_matrix(w1);
  model_weight_matrix(w2);

  computation_graph_compile(graph);

  return graph;
}


static void output_distribution(ComputationGraph *graph, int batch_size) {
  float *output_values = (float *) malloc(sizeof(float) * model_num_classes * batch_size);
  matrix_download(graph->output_node->value, output_values);

  printf("Teddy: output probabilities: [");
  for (int i = 0; i < model_num_classes; i++) {

    if (i > 0){
      printf (", ");
    }

    printf("%.3f", output_values[i * batch_size]);
  }
  printf("]\n");

  int predicted_value = 0;
  float best_confidence = output_values[0];
  for (int i = 1; i < model_num_classes; i++) {
    float confidence = output_values[i * batch_size];
    if (confidence > best_confidence) {
      best_confidence = confidence;
      predicted_value = i;
    }
  }

  printf("Teddy: Predicted digit: %d (%.1f%% confidence)\n", predicted_value, best_confidence * 100.0f);
  free(output_values);
}


int main(int argc, char **argv) {
  const char *kernel_path = "kernel/opencl.c";
  const char *data_directory = "data";

  if (argc > 1) {
    kernel_path = argv[1];
  }
  
  if (argc > 2) {
    data_directory = argv[2];
  }
  printf("\n\033[1mTeddy: A Machine Learning Library in C by AalbatrossGuy (AG).\033[0m\n");
  printf("\033[1mAG: Check out my homelab at https://vargoseus.com/\033[0m\n\n");

  ComputeBackend *teddy_backend = compute_backend_init(kernel_path);
  if (!teddy_backend) {
    fprintf(stderr, "Teddy: Couldn't initialize OpenCL backend. Exiting...\n");
    return 1;
  }

  compute_backend_set_global(teddy_backend);

  char path_buffer[512];

  snprintf(path_buffer, sizeof(path_buffer), "%s/training_images.bin", data_directory);
  float *raw_training_images = load_dataset_binary_f32(path_buffer, model_training_data_count * model_dimm);

  snprintf(path_buffer, sizeof(path_buffer), "%s/training_labels.bin", data_directory);
  float *raw_training_labels = load_dataset_binary_f32(path_buffer, model_training_data_count);

  snprintf(path_buffer, sizeof(path_buffer), "%s/test_images.bin", data_directory);
  float *raw_test_images = load_dataset_binary_f32(path_buffer, model_test_data_count * model_dimm);

  snprintf(path_buffer, sizeof(path_buffer), "%s/test_labels.bin", data_directory);
  float *raw_test_labels = load_dataset_binary_f32(path_buffer, model_test_data_count);

  if (!raw_training_images || !raw_training_labels || !raw_test_images || !raw_test_labels) {
    fprintf(stderr, "Teddy: Failed to load dataset. Download it via the python downloader script.");
    compute_backend_destroy(teddy_backend);
    return 1;
  }

  float *encoded_training_labels = (float *) malloc(sizeof(float) * model_training_data_count * model_num_classes);
  float *encoded_test_labels = (float *) malloc(sizeof(float) * model_test_data_count * model_num_classes);

  one_hot_encode(encoded_training_labels, raw_training_labels, model_training_data_count, model_num_classes);
  one_hot_encode(encoded_test_labels, raw_test_labels, model_test_data_count, model_num_classes);

  srand((unsigned int) time(NULL));
  int demo_sample_index = rand() % model_training_data_count;
  const float *demo_sample_image = raw_training_images + (size_t) demo_sample_index * model_dimm;

  unsigned int rng_seed = 1337u;
  const char *seed_override = getenv("TEDDY_SEED");
  if (seed_override) {
    rng_seed = (unsigned int) strtoul(seed_override, NULL, 10);
  }
  printf("Teddy: RNG seed: %u \n", rng_seed);
  srand(rng_seed);

  printf("\n======== Sample Training Digit ==========\n");
  model_draw_digit(demo_sample_image);
  printf("Teddy: label: %d\n\n", (int) raw_training_labels[demo_sample_index]);

  ComputationGraph *teddy = model_build(model_batch_size);

  printf("\n======== Pre-training Inference ==========\n");
  get_model_prediction(teddy, demo_sample_image, model_dimm, model_batch_size);
  compute_backend_finish(teddy_backend);
  output_distribution(teddy, model_batch_size);

  printf("\n======== Training ===========\n");
  printf("Teddy: training samples: %d | test samples: %d\n", model_training_data_count, model_test_data_count);
  printf("Teddy: batch size: %d | epochs: %d | learning rate: %.3f\n", model_batch_size, model_epochs, model_learning_rate);

  TrainingParams training_parameters = {
    raw_training_images,
    encoded_training_labels,
    raw_test_images,
    encoded_test_labels,
    model_training_data_count,
    model_test_data_count,
    model_dimm,
    model_num_classes,
    model_epochs,
    model_batch_size,
    model_learning_rate
  };

  struct timespec train_start, train_end;
  clock_gettime(CLOCK_MONOTONIC, &train_start);
  train_model(teddy, &training_parameters);
  clock_gettime(CLOCK_MONOTONIC, &train_end);
  compute_backend_finish(teddy_backend);

  double train_seconds = (train_end.tv_sec - train_start.tv_sec) + (train_end.tv_nsec - train_start.tv_nsec) / 1e9;

  printf("\n======== Post-training Inference=======\n");
  get_model_prediction(teddy, demo_sample_image, model_dimm, model_batch_size);
  compute_backend_finish(teddy_backend);
  output_distribution(teddy, model_batch_size);

  printf("\n=========Teddy Evaluation=========\n");
  evaluate_model_prediction(teddy, &training_parameters);

  computation_graph_destroy(teddy);
  free(raw_training_images);
  free(raw_training_labels);
  free(raw_test_images);
  free(raw_test_labels);
  free(encoded_training_labels);
  free(encoded_test_labels);

  compute_backend_destroy(teddy_backend);

  printf("\nTeddy: training took %.2fs\n", train_seconds);
  printf("Teddy: Run finished. Au revoir!\n");

}
