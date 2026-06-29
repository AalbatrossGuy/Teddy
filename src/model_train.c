#include "model_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


void model_weight_matrix(GraphNode *weight_node) {
    int neuron_in_count = weight_node->value->columns;
    int neuron_out_count = weight_node->value->rows;
    float bound = sqrtf(6.0f / (float)(neuron_in_count + neuron_out_count));
    matrix_fill_random(weight_node->value, -bound, bound);
}

void get_model_prediction(ComputationGraph *graph, const float *input_data) {
    matrix_upload(graph->input_node->value, input_data);
    compiled_graph_forward(graph->graph_forward);
}

static void shuffle_samples(int *indices, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}
