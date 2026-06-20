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

