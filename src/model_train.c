#include "model_train.h"
#include "computation_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void model_weight_matrix(GraphNode *weight_node) {
    int neuron_in_count = weight_node->value->columns;
    int neuron_out_count = weight_node->value->rows;
    float bound = sqrtf(6.0f / (float)(neuron_in_count + neuron_out_count));
    matrix_fill_random(weight_node->value, -bound, bound);
}

void get_model_prediction(ComputationGraph *graph, const float *input_data, int in_dim, int batch_size) {
    float *packed_input = (float *)malloc(sizeof(float) * in_dim * batch_size);

    for (int feature = 0; feature < in_dim; feature++) {
        for (int sample = 0; sample < batch_size; sample++) {
            packed_input[feature * batch_size + sample] = input_data[feature];
        }
    }

    matrix_upload(graph->input_node->value, packed_input);
    computation_graph_forward(graph->graph_forward);

    free(packed_input);
}

static void shuffle_samples(int *indices, int count) {
    for (int i = count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

static void pack_batch(float *dst, const float *dataset, const int *indices, int batch_size, int feature_dim) {
    for (int feature = 0; feature < feature_dim; feature++) {
        for (int sample = 0; sample < batch_size; sample++) {
            dst[feature * batch_size + sample] = dataset[indices[sample] * feature_dim + feature];
        }
    }
}

void train_model(ComputationGraph *graph, TrainingParams *model_config) {
    int batch_size = model_config->batch_size;
    int in_dim = model_config->in_dim;
    int out_dim = model_config->out_dim;
    int batches_per_epoch = model_config->training_samples / batch_size;

    int *sample_sequence = (int *)malloc(sizeof(int) * model_config->training_samples);
    for (int i = 0; i < model_config->training_samples; i++)
        sample_sequence[i] = i;

    float *batch_input = (float *)malloc(sizeof(float) * in_dim * batch_size);
    float *batch_target = (float *)malloc(sizeof(float) * out_dim * batch_size);

    for (int epoch = 0; epoch < model_config->epochs; epoch++) {
        shuffle_samples(sample_sequence, model_config->training_samples);

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            CompiledGraph *loss_function = graph->graph_loss;
            for (int n = 0; n < loss_function->length; n++) {
                GraphNode *node = loss_function->ordered_nodes[n];
                if (node->flags & GRAPH_NODE_PARAMETER)
                    matrix_clear(node->gradient);
            }

            const int *batch_indices = sample_sequence + batch * batch_size;
            pack_batch(batch_input, model_config->training_images, batch_indices, batch_size, in_dim);
            pack_batch(batch_target, model_config->training_labels, batch_indices, batch_size, out_dim);

            matrix_upload(graph->input_node->value, batch_input);
            matrix_upload(graph->target_node->value, batch_target);

            computation_graph_forward(loss_function);
            computation_graph_backward(loss_function);

            float average_cost = matrix_sum(graph->loss_node->value) / (float)batch_size;

            float scaled_learning_rate = model_config->lr / (float)batch_size;
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
    free(batch_input);
    free(batch_target);
}

void evaluate_model_prediction(ComputationGraph *graph, TrainingParams *model_config) {
    int batch_size = model_config->batch_size;
    int in_dim = model_config->in_dim;
    int out_dim = model_config->out_dim;
    int batch_count = model_config->test_samples / batch_size;

    int *sequential_indices = (int *)malloc(sizeof(int) * model_config->test_samples);
    for (int i = 0; i < model_config->test_samples; i++)
        sequential_indices[i] = i;

    float *batch_input = (float *)malloc(sizeof(float) * in_dim * batch_size);
    float *batch_target = (float *)malloc(sizeof(float) * out_dim * batch_size);
    float *output_buffer = (float *)malloc(sizeof(float) * out_dim * batch_size);
    float *target_buffer = (float *)malloc(sizeof(float) * out_dim * batch_size);

    int correct_predictions = 0;
    float total_cost = 0.0f;

    for (int batch = 0; batch < batch_count; batch++) {
        const int *batch_indices = sequential_indices + batch * batch_size;
        pack_batch(batch_input, model_config->test_images, batch_indices, batch_size, in_dim);
        pack_batch(batch_target, model_config->test_labels, batch_indices, batch_size, out_dim);

        matrix_upload(graph->input_node->value, batch_input);
        matrix_upload(graph->target_node->value, batch_target);

        computation_graph_forward(graph->graph_loss);

        total_cost += matrix_sum(graph->loss_node->value);

        matrix_download(graph->output_node->value, output_buffer);
        matrix_download(graph->target_node->value, target_buffer);

        for (int sample = 0; sample < batch_size; sample++) {
            int predicted = 0;
            float best_predicted = output_buffer[sample];
            for (int c = 1; c < out_dim; c++) {
                float value = output_buffer[c * batch_size + sample];
                if (value > best_predicted) {
                    best_predicted = value;
                    predicted = c;
                }
            }

            int actual = 0;
            float best_actual = target_buffer[sample];
            for (int c = 1; c < out_dim; c++) {
                float value = target_buffer[c * batch_size + sample];
                if (value > best_actual) {
                    best_actual = value;
                    actual = c;
                }
            }

            if (predicted == actual)
                correct_predictions++;
        }
    }

    int evaluated_samples = batch_count * batch_size;
    float accuracy_percentage = 100.0f * (float)correct_predictions / (float)evaluated_samples;
    float average_cost = total_cost / (float)evaluated_samples;

    printf("Teddy:  Model test results: %d/%d correct (%.2f%%) | average cost: %.4f\n", correct_predictions, evaluated_samples, accuracy_percentage, average_cost);

    free(sequential_indices);
    free(batch_input);
    free(batch_target);
    free(output_buffer);
    free(target_buffer);
}
