// Created by AG on 11-04-2026

#include "computation_engine.h"
#include "matrix_ops.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_NODE_CAPACITY 64

static void ascertain_node_capacity(ComputationGraph *graph) {
  if (graph->node_count < graph->node_capacity) {
    return;
  }

  graph->node_capacity *= 2;
  graph->nodes = (GraphNode **)realloc(graph->nodes, sizeof(GraphNode *) * graph->node_capacity);
}

static GraphNode *allocate_node(ComputationGraph *graph, int rows, int columns, GraphOperationType operation, uint32_t flags) {
  ascertain_node_capacity(graph);

  GraphNode *node = (GraphNode *)calloc(1, sizeof(GraphNode));
  node->index = graph->node_count;
  node->operation = operation;
  node->flags = flags;
  node->value = matrix_create(rows, columns);

  if (flags & GRAPH_NODE_REQUIRES_GRAD) {
    node->gradient = matrix_create(rows, columns);
  }

  graph->nodes[graph->node_count] = node;
  graph->node_count++;

  if (flags & GRAPH_NODE_INPUT) {
    graph->input_node = node;
  }

  if (flags & GRAPH_NODE_OUTPUT) {
    graph->output_node = node;
  }

  if (flags & GRAPH_NODE_TARGET) {
    graph->target_node = node;
  }

  if (flags & GRAPH_NODE_LOSS) {
    graph->loss_node = node;
  }

  return node;
}

static GraphNode *create_unary_node(ComputationGraph *graph, GraphNode *input_node, GraphOperationType operation, uint32_t flags) {
  if (input_node->flags & GRAPH_NODE_REQUIRES_GRAD) {
    flags |= GRAPH_NODE_REQUIRES_GRAD;
  }

  GraphNode *node = allocate_node(graph, input_node->value->rows, input_node->value->columns, operation, flags);
  node->node_inputs[0] = input_node;
  return node;
}

static GraphNode *create_binary_node(ComputationGraph *graph, GraphNode *node_a, GraphNode *node_b, int out_rows, int out_columns, GraphOperationType operation, uint32_t flags) {
  if ((node_a->flags | node_b->flags) & GRAPH_NODE_REQUIRES_GRAD) {
    flags |= GRAPH_NODE_REQUIRES_GRAD;
  }

  GraphNode *node = allocate_node(graph, out_rows, out_columns, operation, flags);
  node->node_inputs[0] = node_a;
  node->node_inputs[1] = node_b;
  return node;
}

ComputationGraph *computation_graph_create(void) {
  ComputationGraph *graph = (ComputationGraph *)calloc(1, sizeof(ComputationGraph));
  graph->node_capacity = INITIAL_NODE_CAPACITY;
  graph->nodes = (GraphNode **)calloc(graph->node_capacity, sizeof(GraphNode *));
  return graph;
}

void computation_graph_destroy(ComputationGraph *graph) {
  if (!graph) {
    return;
  }

  for (int i = 0; i < graph->node_count; i++) {
    GraphNode *node = graph->nodes[i];
    matrix_destroy(node->value);

    if (node->gradient) {
      matrix_destroy(node->gradient);
    }

    free(node);
  }

  free(graph->nodes);

  if (graph->graph_forward) {
    free(graph->graph_forward->ordered_nodes);
    free(graph->graph_forward);
  }

  free(graph);
}

GraphNode *computation_graph_variable(ComputationGraph *graph, int rows, int columns, uint32_t flags) {
  return allocate_node(graph, rows, columns, GRAPH_OP_NONE, flags);
}

GraphNode *computation_graph_reLU(ComputationGraph *graph, GraphNode *input_node, uint32_t flags) {
  return create_unary_node(graph, input_node, GRAPH_OP_RELU, flags);
}

GraphNode *computation_graph_softmax(ComputationGraph *graph, GraphNode *input_node, uint32_t flags) {
  return create_unary_node(graph, input_node, GRAPH_OP_SOFTMAX, flags);
}

GraphNode *computation_graph_add(ComputationGraph *graph, GraphNode *node_a, GraphNode *node_b, uint32_t flags) {
  return create_binary_node(graph, node_a, node_b, node_a->value->rows, node_a->value->columns, GRAPH_OP_ADD, flags);
}

GraphNode *computation_graph_subtract(ComputationGraph *graph, GraphNode *node_a, GraphNode *node_b, uint32_t flags) {
  return create_binary_node(graph, node_a, node_b, node_a->value->rows, node_a->value->columns, GRAPH_OP_SUB, flags);
}

GraphNode *computation_graph_matrix_multiply(ComputationGraph *graph, GraphNode *node_a, GraphNode *node_b, uint32_t flags) {
  return create_binary_node(graph, node_a, node_b, node_a->value->rows, node_b->value->columns, GRAPH_OP_MAT_MUL, flags);
}

GraphNode *computation_graph_cross_entropy(ComputationGraph *graph, GraphNode *predicted_node, GraphNode *expected_node, uint32_t flags) {
  return create_binary_node(graph, predicted_node, expected_node, predicted_node->value->rows, predicted_node->value->columns, GRAPH_OP_CROSS_ENTROPY, flags);
}


static CompiledGraph *sort(ComputationGraph *graph, GraphNode *rootNode) {
    int capacity = graph->node_count;
    int *visited_node = (int *)calloc(capacity, sizeof(int));
    GraphNode **stack = (GraphNode **)malloc(sizeof(GraphNode *) * capacity * 2);
    GraphNode **sorted_output = (GraphNode **)malloc(sizeof(GraphNode *) * capacity);
    int stack_top = 0;
    int output_count = 0;

    stack[stack_top++] = rootNode;

    while (stack_top > 0) {
        GraphNode *current_node = stack[--stack_top];

        if (current_node->index < 0 || current_node->index >= capacity)
            continue;

        if (visited_node[current_node->index]) {
            sorted_output[output_count++] = current_node;
            continue;
        }

        visited_node[current_node->index] = 1;
        stack[stack_top++] = current_node;

        int input_count = graph_op_input_count(current_node->operation);
        for (int i = input_count - 1; i >= 0; i--) {
            GraphNode *dependency = current_node->node_inputs[i];
            if (!dependency) continue;
            if (dependency->index >= 0 && dependency->index < capacity
                && !visited_node[dependency->index]) {
                for (int s = 0; s < stack_top; s++) {
                    if (stack[s] == dependency) {
                        for (int r = s; r < stack_top - 1; r++)
                            stack[r] = stack[r + 1];
                        stack_top--;
                        break;
                    }
                }
                stack[stack_top++] = dependency;
            }
        }
    }

    CompiledGraph *program = malloc(sizeof(CompiledGraph));
    program->length = output_count;
    program->ordered_nodes = (GraphNode **)malloc(sizeof(GraphNode *) * output_count);
    memcpy(program->ordered_nodes, sorted_output, sizeof(GraphNode *) * output_count);

    free(visited_node);
    free(stack);
    free(sorted_output);

    return program;
}

void computation_graph_compile(ComputationGraph *graph) {
    if (graph->output_node)
        graph->graph_forward = sort(graph, graph->output_node);

    if (graph->loss_node)
        graph->graph_loss = sort(graph, graph->loss_node);
}

void computation_graph_forward(CompiledGraph *graph) {
    for (int i = 0; i < graph->length; i++) {
        GraphNode *current_node = graph->ordered_nodes[i];
        GraphNode *input_a = current_node->node_inputs[0];
        GraphNode *input_b = current_node->node_inputs[1];

        switch (current_node->operation) {
            case GRAPH_OP_NONE:
            case GRAPH_OP_UNARY_BEGIN:
            case GRAPH_OP_BINARY_BEGIN:
                break;

            case GRAPH_OP_RELU:
                matrix_reLU(current_node->value, input_a->value);
                break;

            case GRAPH_OP_SOFTMAX:
                matrix_softmax(current_node->value, input_a->value);
                break;

            case GRAPH_OP_ADD:
                matrix_add(current_node->value, input_a->value, input_b->value);
                break;

            case GRAPH_OP_SUB:
                matrix_sub(current_node->value, input_a->value, input_b->value);
                break;

            case GRAPH_OP_MAT_MUL:
                matrix_multiply(current_node->value, input_a->value, input_b->value,
                                0, 0, 1);
                break;

            case GRAPH_OP_CROSS_ENTROPY:
                matrix_cross_entropy(current_node->value, input_a->value, input_b->value);
                break;
        }
    }
}
