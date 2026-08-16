/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_GRAPH_INTERNAL_H
#define M3_GRAPH_INTERNAL_H

#include "m3_op.h"

typedef struct m3_coreml_partition m3_coreml_partition;

typedef struct {
    m3_graph_value_desc description;
    m3_tensor_metadata metadata;
    void *constant_data;
    size_t constant_size;
    bool constant_set;
} m3_graph_value;

typedef struct {
    char *compiled_model_path;
    m3_graph_value_id *inputs;
    char **input_names;
    size_t input_count;
    m3_graph_value_id *outputs;
    char **output_names;
    size_t output_count;
} m3_graph_coreml;

typedef enum {
    M3_GRAPH_NODE_REGULAR = 0,
    M3_GRAPH_NODE_COREML
} m3_graph_node_type;

typedef struct {
    m3_graph_node_type type;
    union {
        m3_graph_node_desc regular;
        m3_graph_coreml coreml;
    } payload;
} m3_graph_node;

struct m3_graph {
    m3_graph_value *values;
    size_t value_count;
    size_t value_capacity;
    m3_graph_node *nodes;
    size_t node_count;
    size_t node_capacity;
};

typedef struct {
    m3_graph_value_desc description;
    m3_tensor_metadata metadata;
    size_t slot;
} m3_session_value;

typedef struct {
    m3_storage *storage;
    size_t byte_count;
} m3_session_slot;

typedef struct {
    m3_graph_node_type type;
    m3_provider_kind provider;
    union {
        size_t regular_index;
        m3_coreml_partition *coreml;
    } payload;
} m3_session_node;

struct m3_session {
    m3_backend *backend;
    m3_session_options options;
    m3_session_value *values;
    m3_tensor_view *views;
    size_t value_count;
    m3_session_slot *slots;
    size_t slot_count;
    m3_session_node *nodes;
    m3_command *commands;
    size_t node_count;
    void *scratch;
    size_t scratch_capacity;
    size_t allocated_bytes;
};

m3_status m3_graph_bind_command(const m3_graph_node_desc *description,
                                m3_tensor_view *views, size_t value_count,
                                m3_command *command, m3_error *error);
m3_status m3_coreml_partition_create(
    const m3_graph_coreml *description, const m3_tensor_view *views,
    size_t value_count, m3_coreml_partition **partition,
    m3_error *error);
void m3_coreml_partition_free(m3_coreml_partition *partition);
m3_status m3_coreml_partition_execute(
    m3_coreml_partition *partition, m3_tensor_view *views,
    size_t value_count, m3_error *error);

#endif
