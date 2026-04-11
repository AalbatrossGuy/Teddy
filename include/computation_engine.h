// Created by AG on 11-04-2026

#ifndef COMPUTATION_ENGINE_H
#define COMPUTATION_ENGINE_H

#include "matrix_ops.h"
#include <stdint.h>

#define MAXIMUM_NODE_INPUTS 2

typedef enum {
  GRAPH_OP_NONE = 0,
  GRAPH_OP_UNARY_BEGIN,
  GRAPH_OP_RELU,
  GRAPH_OP_SOFTMAX,
  GRAPH_OP_BINARY_BEGIN,
  GRAPH_OP_ADD,
  GRAPH_OP_SUB,
  GRAPH_OP_MAT_MUL,
  GRAPH_OP_CROSS_ENTROPY
} GraphOperationType;

typedef enum {
  GRAPH_NODE_NONE = 0,
  GRAPH_NODE_REQUIRES_GRAD = 1 << 0,
  GRAPH_NODE_PARAMETER = 1 << 1,
  GRAPH_NODE_INPUT = 1 << 2,
  GRAPH_NODE_OUTPUT = 1 << 3,
  GRAPH_NODE_TARGET = 1 << 4,
  GRAPH_NODE_LOSS = 1 << 5 
} GraphNodeFlags;

typedef struct GraphNode {
  int index;
  Matrix *value;
  Matrix *gradient;
  GraphOperationType operation;
  uint32_t flags;
  struct GraphNode *node_inputs[MAXIMUM_NODE_INPUTS];
} GraphNode;

typedef struct {
  int length;
  GraphNode **ordered_nodes;
} CompiledGraph;

typedef struct {
  int node_count;
  int node_capacity;
  GraphNode **nodes;
  GraphNode *input_node;
  GraphNode *output_node;
  GraphNode *target_node;
  GraphNode *loss_node;
  CompiledGraph *graph_forward;
  CompiledGraph *graph_loss;
} ComputationGraph;

static inline int graph_op_input_count(GraphOperationType op) {
  if (op <= GRAPH_OP_NONE) {
    return 0;
  }

  if (op < GRAPH_OP_BINARY_BEGIN) {
    return 1;
  }

  return 2;
}

ComputationGraph *computation_graph_create(void);
void computation_graph_destroy(ComputationGraph *graph);

GraphNode *computation_graph_variable(ComputationGraph *graph, int rows, int columns, uint32_t flags);
GraphNode *computation_graph_reLU(ComputationGraph *graph, GraphNode *input_node, uint32_t flags);
GraphNode *computation_graph_softmax(ComputationGraph *graph, GraphNode *input_node, uint32_t flags);
GraphNode *computation_graph_add(ComputationGraph *graph, GraphNode *node_a, GraphNode *node_b, uint32_t flags);
GraphNode *computation_graph_subtract(ComputationGraph *graph, GraphNode *node_a, GraphNode *node_b, uint32_t flags);
GraphNode *computation_graph_matrix_multiply(ComputationGraph *graph, GraphNode *node_a, GraphNode *node_b, uint32_t flags);
GraphNode *computation_graph_cross_entropy(ComputationGraph *graph, GraphNode *predicted_node, GraphNode *expected_node, uint32_t flags);

void computation_graph_compile(ComputationGraph *graph);
void compiled_graph_forward(CompiledGraph *compiled_graph);
void compiled_graph_backward(CompiledGraph *compiled_graph);

#endif
