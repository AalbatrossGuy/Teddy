// Created by AG on 13-08-2026
#include "compute_backend.h"
#include "matrix_ops.h"
#include "computation_engine.h"
#include "model_train.h"
#include "dataset_ops.h"

#include <stdio.h>
#include <stdlib.h>

// Model Defitions
#define model_dimm 784
#define model_num_classes 10
#define model_training_data_count 60000
#define model_test_data_count 10000
#define model_hidden_layer 16

static void model_draw_digit(const float *pixels) {
  for (int row = 0; row < 28; row++) {
    for (int column = 0; column < 28; column++) {
      int grayscale_image_data = (int) pixels[row * 28 + column] * 23.0f;
      printf("\033[48;5;%dm \033[0m", 232 +  grayscale_image_data);
    }
    printf("\n");
  }
}

static ComputationGraph *model_build(void) {
  ComputationGraph *graph = computation_graph_create();
  GraphNode *input_node = computation_graph_variable(graph, model_dimm, 1, GRAPH_NODE_INPUT);
  GraphNode *initial_weight = computation_graph_variable(graph, model_hidden_layer, model_dimm, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *initial_bias_value = computation_graph_variable(graph, model_hidden_layer, 1, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *initial_preactivation_value = computation_graph_matrix_multiply(graph, initial_weight, input_node, 0);
  GraphNode *biased_value = computation_graph_add(graph, initial_preactivation_value, initial_bias_value, 0);
  GraphNode *activation_value = computation_graph_reLU(graph, biased_value, 0);

  GraphNode *w1 = computation_graph_variable(graph, model_hidden_layer, model_hidden_layer, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *bias1 = computation_graph_variable(graph, model_hidden_layer, 1, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *preactivation_value1 = computation_graph_matrix_multiply(graph, w1, activation_value, 0);
  GraphNode *biased_value_1 = computation_graph_add(graph, preactivation_value1, bias1, 0);
  GraphNode *preresidual1 = computation_graph_reLU(graph, biased_value_1, 0);

  GraphNode *residual_sum_value = computation_graph_add(graph, preresidual1, activation_value, 0);

  GraphNode *w2 = computation_graph_variable(graph, model_num_classes, model_hidden_layer, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *bias2 = computation_graph_variable(graph, model_num_classes, 1, GRAPH_NODE_REQUIRES_GRAD | GRAPH_NODE_PARAMETER);
  GraphNode *preactivation_value2 = computation_graph_matrix_multiply(graph, w2, residual_sum_value, 0);
  GraphNode *biased_value_2 = computation_graph_add(graph, preactivation_value2, bias2, 0);
  
  GraphNode *output = computation_graph_softmax(graph, biased_value_2, GRAPH_NODE_OUTPUT);
  GraphNode *target_value = computation_graph_variable(graph, model_num_classes, 1, GRAPH_NODE_TARGET);
  GraphNode *loss_value = computation_graph_cross_entropy(graph, output, target_value, GRAPH_NODE_LOSS);


  (void) loss_value;
  model_weight_matrix(initial_weight);
  model_weight_matrix(w1);
  model_weight_matrix(w2);

  computation_graph_compile(graph);

  return graph;
}


static void output_distribution(ComputationGraph *graph) {
  float output_values[model_num_classes]; // 10
  matrix_download(graph->output_node->value, output_values);

  printf("Teddy: output probabilities: [");
  for (int i = 0; i < model_num_classes; i++) {

    if (i > 0){
      printf (", ");
    }

    printf("%.3f", output_values[i]);
  }
  printf("]\n");

  int predicted_value = matrix_argmax(graph->output_node->value);
  printf("Teddy: Predicted digit: %d (%.1f%% confidence)\n", predicted_value, output_values[predicted_value] * 100.0f);
}


int main(int argc, char **argv) {
  const char *kernel_path = "kernels/opencl.cl";
  const char *data_directory = "data";

  if (argc > 1) {
    kernel_path = argv[1];
  }
  
  if (argc > 2) {
    data_directory = argv[2];
  }
  printf("Teddy.");

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
  float *raw_test_images = load_dataset_binary_f32(path_buffer, model_training_data_count * model_dimm);

  snprintf(path_buffer, sizeof(path_buffer), "%s/test_labels.bin", data_directory);
  float *raw_test_labels = load_dataset_binary_f32(path_buffer, model_training_data_count);

  if (!raw_training_images || !raw_training_labels || !raw_test_images || !raw_test_labels) {
    fprintf(stderr, "Teddy: Failed to load dataset. Download it via the python downloader script.");
    compute_backend_destroy(teddy_backend);
    return 1;
  }

  float *encoded_training_labels = (float *) malloc(sizeof(float) * model_training_data_count * model_num_classes);
  float *encoded_test_labels = (float *) malloc(sizeof(float) * model_training_data_count * model_num_classes);

  one_hot_encode(encoded_training_labels, raw_training_labels, model_training_data_count, model_num_classes);
  one_hot_encode(encoded_test_labels, raw_test_labels, model_training_data_count, model_num_classes);

  printf("\n======== Sample Training Digit ==========\n");
  model_draw_digit(raw_training_images);
  printf("Teddy: label: %d\n\n", (int) raw_training_labels[0]);

  ComputationGraph *teddy = model_build();

  printf("\n======== Pre-training Inference ==========\n");
  get_model_prediction(teddy, raw_training_images);
  compute_backend_finish(teddy_backend);
  output_distribution(teddy);

  printf("\n======== Training ===========\n");
  TrainingParams training_parameters = {
    raw_training_images,
    encoded_training_labels,
    raw_test_images,
    encoded_test_labels,
    model_training_data_count,
    model_test_data_count,
    model_dimm,
    model_num_classes,
    3,
    50,
    0.05f
  };

  train_model(teddy, &training_parameters);
  compute_backend_finish(teddy_backend);

  printf("\n======== Post-training Inference=======\n");
  get_model_prediction(teddy, raw_training_images);
  compute_backend_finish(teddy_backend);
  output_distribution(teddy);

  printf("=========Teddy Evaluation=========");
  evaluate_model_prediction(teddy, &training_parameters);

  computation_graph_destroy(teddy);
  free(raw_training_images);
  free(raw_training_labels);
  free(raw_test_images);
  free(raw_test_labels);
  free(encoded_training_labels);
  free(encoded_test_labels);

  compute_backend_destroy(teddy_backend);

  printf("\nTeddy: Run finished. Au revoir!\n");

}
