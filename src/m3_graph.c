/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_graph_internal.h"

#include <stdlib.h>
#include <string.h>

static char *m3_graph_string_copy(const char *source)
{
    size_t length;
    char *copy;

    if (source == NULL) {
        return NULL;
    }
    length = strlen(source);
    if (length == SIZE_MAX) {
        return NULL;
    }
    copy = malloc(length + 1U);
    if (copy != NULL) {
        (void)memcpy(copy, source, length + 1U);
    }
    return copy;
}

void m3_graph_value_desc_init(m3_graph_value_desc *description,
                              m3_graph_value_role role, m3_dtype dtype,
                              uint8_t rank, const uint64_t *shape)
{
    uint8_t axis;

    if (description == NULL) {
        return;
    }
    (void)memset(description, 0, sizeof(*description));
    description->role = role;
    description->dtype = dtype;
    description->rank = rank;
    if (shape != NULL) {
        for (axis = 0U; axis < rank && axis < M3_TENSOR_MAX_RANK; ++axis) {
            description->shape[axis] = shape[axis];
        }
    }
}

void m3_graph_node_desc_init(m3_graph_node_desc *description,
                             m3_op_kind kind)
{
    size_t index;

    if (description == NULL) {
        return;
    }
    (void)memset(description, 0, sizeof(*description));
    description->kind = kind;
    for (index = 0U; index < M3_GRAPH_NODE_INPUT_CAPACITY; ++index) {
        description->inputs[index] = M3_GRAPH_VALUE_NONE;
    }
    for (index = 0U; index < M3_GRAPH_NODE_OUTPUT_CAPACITY; ++index) {
        description->outputs[index] = M3_GRAPH_VALUE_NONE;
    }
}

void m3_session_options_init(m3_session_options *options)
{
    if (options != NULL) {
        options->allow_neural_engine = true;
        options->allow_metal = true;
        options->allow_cpu = true;
    }
}

m3_status m3_graph_create(m3_graph **graph, m3_error *error)
{
    m3_graph *created;

    if (graph == NULL || *graph != NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "graph output must be empty");
    }
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate graph");
    }
    *graph = created;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static void m3_graph_coreml_dispose(m3_graph_coreml *coreml)
{
    size_t index;

    if (coreml == NULL) {
        return;
    }
    free(coreml->compiled_model_path);
    for (index = 0U; coreml->input_names != NULL &&
                     index < coreml->input_count; ++index) {
        free(coreml->input_names[index]);
    }
    for (index = 0U; coreml->output_names != NULL &&
                     index < coreml->output_count; ++index) {
        free(coreml->output_names[index]);
    }
    free(coreml->inputs);
    free(coreml->input_names);
    free(coreml->outputs);
    free(coreml->output_names);
    (void)memset(coreml, 0, sizeof(*coreml));
}

void m3_graph_free(m3_graph *graph)
{
    size_t index;

    if (graph == NULL) {
        return;
    }
    for (index = 0U; index < graph->value_count; ++index) {
        free(graph->values[index].constant_data);
    }
    for (index = 0U; index < graph->node_count; ++index) {
        if (graph->nodes[index].type == M3_GRAPH_NODE_COREML) {
            m3_graph_coreml_dispose(&graph->nodes[index].payload.coreml);
        }
    }
    free(graph->values);
    free(graph->nodes);
    free(graph);
}

static bool m3_graph_grow(void **memory, size_t *capacity,
                          size_t element_size)
{
    size_t next = *capacity == 0U ? 8U : *capacity * 2U;
    void *replacement;

    if (next < *capacity || next > SIZE_MAX / element_size) {
        return false;
    }
    replacement = realloc(*memory, next * element_size);
    if (replacement == NULL) {
        return false;
    }
    *memory = replacement;
    *capacity = next;
    return true;
}

m3_status m3_graph_add_value(m3_graph *graph,
                             const m3_graph_value_desc *description,
                             m3_graph_value_id *value, m3_error *error)
{
    m3_graph_value added;
    m3_status status;

    if (graph == NULL || description == NULL || value == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "graph, value description, and output are required");
    }
    if (description->role < M3_GRAPH_INPUT ||
        description->role > M3_GRAPH_OUTPUT) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "graph value role is invalid");
    }
    if (graph->value_count >= (size_t)M3_GRAPH_VALUE_NONE) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "graph value identifiers are exhausted");
    }
    (void)memset(&added, 0, sizeof(added));
    status = m3_tensor_metadata_init(&added.metadata, description->dtype,
                                     description->rank, description->shape,
                                     error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    added.description = *description;
    if (graph->value_count == graph->value_capacity &&
        !m3_graph_grow((void **)&graph->values, &graph->value_capacity,
                       sizeof(*graph->values))) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot grow graph values");
    }
    *value = (m3_graph_value_id)graph->value_count;
    graph->values[graph->value_count] = added;
    ++graph->value_count;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_graph_set_constant(m3_graph *graph,
                                m3_graph_value_id value,
                                const void *data, size_t byte_count,
                                m3_error *error)
{
    m3_graph_value *target;
    void *copy = NULL;

    if (graph == NULL || (size_t)value >= graph->value_count) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "constant graph value is out of range");
    }
    target = &graph->values[value];
    if (target->description.role != M3_GRAPH_CONSTANT ||
        byte_count != target->metadata.byte_count ||
        (byte_count != 0U && data == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "constant data does not match its graph value");
    }
    if (byte_count != 0U) {
        copy = malloc(byte_count);
        if (copy == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot copy graph constant");
        }
        (void)memcpy(copy, data, byte_count);
    }
    free(target->constant_data);
    target->constant_data = copy;
    target->constant_size = byte_count;
    target->constant_set = true;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static m3_status m3_graph_append_node(m3_graph *graph,
                                      const m3_graph_node *node,
                                      m3_error *error)
{
    if (graph->node_count == graph->node_capacity &&
        !m3_graph_grow((void **)&graph->nodes, &graph->node_capacity,
                       sizeof(*graph->nodes))) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot grow graph nodes");
    }
    graph->nodes[graph->node_count] = *node;
    ++graph->node_count;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_graph_add_node(m3_graph *graph,
                            const m3_graph_node_desc *description,
                            m3_error *error)
{
    m3_graph_node node;

    if (graph == NULL || description == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "graph and node description are required");
    }
    (void)memset(&node, 0, sizeof(node));
    node.type = M3_GRAPH_NODE_REGULAR;
    node.payload.regular = *description;
    return m3_graph_append_node(graph, &node, error);
}

static m3_status m3_graph_coreml_copy_side(
    const m3_graph_value_id *values, const char *const *names, size_t count,
    m3_graph_value_id **copied_values, char ***copied_names,
    m3_error *error)
{
    size_t index;

    *copied_values = calloc(count, sizeof(**copied_values));
    *copied_names = calloc(count, sizeof(**copied_names));
    if (*copied_values == NULL || *copied_names == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate Core ML feature bindings");
    }
    for (index = 0U; index < count; ++index) {
        if (names[index] == NULL || names[index][0] == '\0') {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "Core ML feature name is empty");
        }
        (*copied_values)[index] = values[index];
        (*copied_names)[index] = m3_graph_string_copy(names[index]);
        if ((*copied_names)[index] == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot copy Core ML feature name");
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_graph_add_coreml_partition(
    m3_graph *graph, const m3_coreml_partition_desc *description,
    m3_error *error)
{
    m3_graph_node node;
    m3_status status;

    if (graph == NULL || description == NULL ||
        description->compiled_model_path == NULL ||
        description->compiled_model_path[0] == '\0' ||
        description->input_count == 0U ||
        description->output_count == 0U ||
        description->inputs == NULL || description->input_names == NULL ||
        description->outputs == NULL || description->output_names == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Core ML partition description is incomplete");
    }
    (void)memset(&node, 0, sizeof(node));
    node.type = M3_GRAPH_NODE_COREML;
    node.payload.coreml.compiled_model_path =
        m3_graph_string_copy(description->compiled_model_path);
    if (node.payload.coreml.compiled_model_path == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot copy Core ML model path");
    }
    node.payload.coreml.input_count = description->input_count;
    node.payload.coreml.output_count = description->output_count;
    status = m3_graph_coreml_copy_side(
        description->inputs, description->input_names,
        description->input_count, &node.payload.coreml.inputs,
        &node.payload.coreml.input_names, error);
    if (status == M3_STATUS_OK) {
        status = m3_graph_coreml_copy_side(
            description->outputs, description->output_names,
            description->output_count, &node.payload.coreml.outputs,
            &node.payload.coreml.output_names, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_append_node(graph, &node, error);
    }
    if (status != M3_STATUS_OK) {
        m3_graph_coreml_dispose(&node.payload.coreml);
    }
    return status;
}
