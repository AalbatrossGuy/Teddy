#ifndef MODEL_TRAIN_H
#define MODEL_TRAIN_H

#include "computation_engine.h"

typedef struct {
  float *training_images;
  float *training_labels;

  float *test_images;
  float *test_labels;

  int training_samples;
  int test_samples;
  int in_dim;
  int out_dim;
  int epochs;
  int batch_size;
  float lr;
} TrainingParams;

void get_model_prediction(ComputationGraph *graph, const float *input_data);
void train_model(ComputationGraph *graph, TrainingParams *config);
void evaluate_model_prediction(ComputationGraph *graph, TrainingParams *config);
void model_weight_matrix(GraphNode *weight_node);

#endif // !MODEL_TRAIN_H
