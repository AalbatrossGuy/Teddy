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

void train_model(ComputationGraph *graph, TrainingParams *model_config) {
    srand((unsigned int)time(NULL));

    int batches_per_epoch = model_config->training_samples / model_config->batch_size;

    int *sample_sequence = (int *)malloc(sizeof(int) * model_config->training_samples);
    for (int i = 0; i < model_config->training_samples; i++)
        sample_sequence[i] = i;

    float *sample_input = (float *)malloc(sizeof(float) * model_config->in_dim);
    float *sample_target_label = (float *)malloc(sizeof(float) * model_config->out_dim);

    for (int epoch = 0; epoch < model_config->epochs; epoch++) {
        shuffle_samples(sample_sequence, model_config->training_samples);

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            CompiledGraph *loss_function = graph->graph_loss;
            for (int n = 0; n < loss_function->length; n++) {
                GraphNode *node = loss_function->ordered_nodes[n];
                if (node->flags & GRAPH_NODE_PARAMETER)
                    matrix_clear(node->gradient);
            }

            float batch_cost = 0.0f;

            for (int sample = 0; sample < model_config->batch_size; sample++) {
                int sample_index = sample_sequence[batch * model_config->batch_size + sample];

                memcpy(sample_input, model_config->training_images + sample_index * model_config->in_dim, sizeof(float) * model_config->in_dim);
                memcpy(sample_target_label, model_config->training_labels + sample_index * model_config->out_dim, sizeof(float) * model_config->out_dim);

                matrix_upload(graph->input_node->value, sample_input);
                matrix_upload(graph->target_node->value, sample_target_label);

                compiled_graph_forward(loss_function);
                compiled_graph_backward(loss_function);

                batch_cost += matrix_sum(graph->loss_node->value);
            }

            float average_cost = batch_cost / (float)model_config->batch_size;

            float scaled_learning_rate = model_config->lr / (float)model_config->batch_size;
            for (int n = 0; n < loss_function->length; n++) {
                GraphNode *node = loss_function->ordered_nodes[n];
                if (!(node->flags & GRAPH_NODE_PARAMETER))
                    continue;
                matrix_param_update(node->value, node->gradient, scaled_learning_rate);
            }

            compute_backend_finish(compute_backend_global());

            printf("\rTeddy:  epoch %2d/%d | batch %4d/%d | cost %.4f", epoch + 1, model_config->epochs, batch + 1, batches_per_epoch, average_cost);
            fflush(stdout);
        }
        printf("\n");
    }

    free(sample_sequence);
    free(sample_input);
    free(sample_target_label);
}
