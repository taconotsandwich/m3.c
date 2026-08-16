/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_graph_internal.h"

#include "m3_provider.h"

#include <stdlib.h>
#include <string.h>

#define M3_GRAPH_STORAGE_ALIGNMENT 64U

typedef struct {
    size_t input_count;
    size_t output_count;
    uint32_t optional_inputs;
} m3_graph_schema;

static m3_status m3_graph_regular_schema(m3_op_kind kind,
                                          m3_graph_schema *schema,
                                          m3_error *error)
{
    (void)memset(schema, 0, sizeof(*schema));
    schema->output_count = 1U;
    switch (kind) {
    case M3_OP_COPY:
    case M3_OP_CAST:
    case M3_OP_SOFTMAX:
    case M3_OP_NEAREST_RESIZE1D:
    case M3_OP_TANH:
        schema->input_count = 1U;
        break;
    case M3_OP_ADD:
    case M3_OP_MUL:
    case M3_OP_EMBEDDING:
    case M3_OP_MATMUL:
    case M3_OP_RMS_NORM:
    case M3_OP_GATED_SILU:
    case M3_OP_CATEGORICAL:
    case M3_OP_SNAKE1D:
        schema->input_count = 2U;
        break;
    case M3_OP_LINEAR:
    case M3_OP_LAYER_NORM:
    case M3_OP_CONV1D:
    case M3_OP_CONV_TRANSPOSE1D:
        schema->input_count = 3U;
        schema->optional_inputs = 1U << 2U;
        break;
    case M3_OP_ROPE:
        schema->input_count = 3U;
        break;
    case M3_OP_ATTENTION:
        schema->input_count = 4U;
        schema->optional_inputs = 1U << 3U;
        break;
    case M3_OP_TOP_K:
        schema->input_count = 1U;
        schema->output_count = 2U;
        break;
    default:
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "unsupported graph operation kind %d",
                            (int)kind);
    }
    return M3_STATUS_OK;
}

static m3_status m3_graph_check_value_id(
    const m3_graph *graph, m3_graph_value_id value, bool optional,
    const char *kind, m3_error *error)
{
    if (value == M3_GRAPH_VALUE_NONE) {
        if (optional) {
            return M3_STATUS_OK;
        }
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "required graph node %s is absent", kind);
    }
    if ((size_t)value >= graph->value_count) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "graph node %s is out of range", kind);
    }
    return M3_STATUS_OK;
}

static m3_status m3_graph_regular_ids(
    const m3_graph *graph, const m3_graph_node_desc *node,
    m3_graph_schema *schema, m3_error *error)
{
    size_t index;
    m3_status status = m3_graph_regular_schema(node->kind, schema, error);

    for (index = 0U; status == M3_STATUS_OK &&
                       index < M3_GRAPH_NODE_INPUT_CAPACITY; ++index) {
        bool used = index < schema->input_count;
        bool optional = used &&
            (schema->optional_inputs & (1U << (uint32_t)index)) != 0U;

        if (!used && node->inputs[index] != M3_GRAPH_VALUE_NONE) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "graph node has an extra input");
        } else if (used) {
            status = m3_graph_check_value_id(
                graph, node->inputs[index], optional, "input", error);
        }
    }
    for (index = 0U; status == M3_STATUS_OK &&
                       index < M3_GRAPH_NODE_OUTPUT_CAPACITY; ++index) {
        bool used = index < schema->output_count;

        if (!used && node->outputs[index] != M3_GRAPH_VALUE_NONE) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "graph node has an extra output");
        } else if (used) {
            status = m3_graph_check_value_id(
                graph, node->outputs[index], false, "output", error);
        }
    }
    return status;
}

static m3_status m3_graph_topology(const m3_graph *graph,
                                    size_t *producers,
                                    size_t *last_uses,
                                    m3_error *error)
{
    bool *available;
    size_t node_index;
    size_t value_index;
    m3_status status = M3_STATUS_OK;

    available = calloc(graph->value_count, sizeof(*available));
    if (graph->value_count != 0U && available == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate graph topology state");
    }
    for (value_index = 0U; value_index < graph->value_count; ++value_index) {
        const m3_graph_value *value = &graph->values[value_index];

        producers[value_index] = SIZE_MAX;
        last_uses[value_index] = 0U;
        available[value_index] =
            value->description.role == M3_GRAPH_INPUT ||
            value->description.role == M3_GRAPH_CONSTANT;
        if (value->description.role == M3_GRAPH_CONSTANT &&
            !value->constant_set) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "graph constant has no data");
            break;
        }
    }
    for (node_index = 0U; status == M3_STATUS_OK &&
                          node_index < graph->node_count; ++node_index) {
        const m3_graph_node *node = &graph->nodes[node_index];
        const m3_graph_value_id *inputs;
        const m3_graph_value_id *outputs;
        size_t input_count;
        size_t output_count;
        size_t index;
        m3_graph_schema schema;

        if (node->type == M3_GRAPH_NODE_REGULAR) {
            status = m3_graph_regular_ids(
                graph, &node->payload.regular, &schema, error);
            inputs = node->payload.regular.inputs;
            outputs = node->payload.regular.outputs;
            input_count = schema.input_count;
            output_count = schema.output_count;
        } else {
            const m3_graph_coreml *coreml = &node->payload.coreml;

            inputs = coreml->inputs;
            outputs = coreml->outputs;
            input_count = coreml->input_count;
            output_count = coreml->output_count;
            for (index = 0U; status == M3_STATUS_OK &&
                               index < input_count; ++index) {
                status = m3_graph_check_value_id(
                    graph, inputs[index], false, "Core ML input", error);
            }
            for (index = 0U; status == M3_STATUS_OK &&
                               index < output_count; ++index) {
                status = m3_graph_check_value_id(
                    graph, outputs[index], false, "Core ML output", error);
            }
        }
        for (index = 0U; status == M3_STATUS_OK &&
                           index < input_count; ++index) {
            m3_graph_value_id input = inputs[index];

            if (input == M3_GRAPH_VALUE_NONE) {
                continue;
            }
            if (!available[input]) {
                status = m3_error_set(
                    error, M3_STATUS_INVALID_ARGUMENT,
                    "graph node reads a value before it is produced");
            } else {
                last_uses[input] = node_index;
            }
        }
        for (index = 0U; status == M3_STATUS_OK &&
                           index < output_count; ++index) {
            m3_graph_value_id output = outputs[index];
            m3_graph_value_role role =
                graph->values[output].description.role;
            size_t other;

            if (role != M3_GRAPH_TEMPORARY && role != M3_GRAPH_OUTPUT) {
                status = m3_error_set(
                    error, M3_STATUS_INVALID_ARGUMENT,
                    "graph node writes an input or constant value");
            } else if (producers[output] != SIZE_MAX) {
                status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                      "graph value has multiple producers");
            }
            for (other = 0U; status == M3_STATUS_OK && other < index;
                 ++other) {
                if (outputs[other] == output) {
                    status = m3_error_set(
                        error, M3_STATUS_INVALID_ARGUMENT,
                        "graph node repeats an output value");
                }
            }
            if (status == M3_STATUS_OK) {
                producers[output] = node_index;
                last_uses[output] = node_index;
                available[output] = true;
            }
        }
    }
    for (value_index = 0U; status == M3_STATUS_OK &&
                           value_index < graph->value_count; ++value_index) {
        m3_graph_value_role role =
            graph->values[value_index].description.role;

        if ((role == M3_GRAPH_TEMPORARY || role == M3_GRAPH_OUTPUT) &&
            producers[value_index] == SIZE_MAX) {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "graph value has no producer");
        }
    }
    free(available);
    return status;
}

static m3_status m3_graph_add_slot(m3_session *session, size_t byte_count,
                                    size_t *slot, m3_error *error)
{
    if (session->slot_count >= session->value_count) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "graph storage slot capacity is exhausted");
    }
    *slot = session->slot_count;
    session->slots[*slot].byte_count = byte_count;
    ++session->slot_count;
    return M3_STATUS_OK;
}

static m3_status m3_graph_plan_slots(m3_session *session,
                                     const size_t *producers,
                                     const size_t *last_uses,
                                     m3_error *error)
{
    size_t *slot_last_use;
    size_t value_index;
    size_t node_index;
    m3_status status = M3_STATUS_OK;

    slot_last_use = calloc(session->value_count, sizeof(*slot_last_use));
    if (session->value_count != 0U && slot_last_use == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate liveness slots");
    }
    for (value_index = 0U; value_index < session->value_count;
         ++value_index) {
        m3_session_value *value = &session->values[value_index];

        value->slot = SIZE_MAX;
        if (value->description.role != M3_GRAPH_TEMPORARY) {
            status = m3_graph_add_slot(
                session, value->metadata.byte_count, &value->slot, error);
            if (status != M3_STATUS_OK) {
                break;
            }
            slot_last_use[value->slot] = SIZE_MAX;
        }
    }
    for (node_index = 0U; status == M3_STATUS_OK &&
                          node_index < session->node_count; ++node_index) {
        for (value_index = 0U; value_index < session->value_count;
             ++value_index) {
            m3_session_value *value = &session->values[value_index];
            size_t best = SIZE_MAX;
            size_t slot;

            if (value->description.role != M3_GRAPH_TEMPORARY ||
                producers[value_index] != node_index) {
                continue;
            }
            for (slot = 0U; slot < session->slot_count; ++slot) {
                if (slot_last_use[slot] < node_index &&
                    session->slots[slot].byte_count >=
                        value->metadata.byte_count &&
                    (best == SIZE_MAX ||
                     session->slots[slot].byte_count <
                         session->slots[best].byte_count)) {
                    best = slot;
                }
            }
            if (best == SIZE_MAX) {
                status = m3_graph_add_slot(
                    session, value->metadata.byte_count, &best, error);
                if (status != M3_STATUS_OK) {
                    break;
                }
            }
            value->slot = best;
            slot_last_use[best] = last_uses[value_index];
        }
    }
    free(slot_last_use);
    return status;
}

static m3_status m3_graph_allocate_slots(m3_session *session,
                                          m3_error *error)
{
    size_t slot;
    m3_status status = M3_STATUS_OK;

    for (slot = 0U; slot < session->slot_count; ++slot) {
        size_t byte_count = session->slots[slot].byte_count;

        if (byte_count > SIZE_MAX - session->allocated_bytes) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "graph allocation total overflows");
        }
        status = m3_storage_allocate(
            session->backend, byte_count, M3_GRAPH_STORAGE_ALIGNMENT,
            &session->slots[slot].storage, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        session->allocated_bytes += byte_count;
    }
    return status;
}

static m3_status m3_graph_build_views(m3_session *session,
                                       const m3_graph *graph,
                                       m3_error *error)
{
    size_t index;

    for (index = 0U; index < session->value_count; ++index) {
        const m3_session_value *value = &session->values[index];
        m3_status status = m3_tensor_view_contiguous(
            &session->views[index], session->slots[value->slot].storage,
            value->metadata.dtype, value->metadata.rank,
            value->metadata.shape, 0U, error);

        if (status != M3_STATUS_OK) {
            return status;
        }
        if (value->description.role == M3_GRAPH_CONSTANT) {
            status = m3_storage_write(
                session->views[index].storage, 0U,
                graph->values[index].constant_data,
                graph->values[index].constant_size, error);
            if (status != M3_STATUS_OK) {
                return status;
            }
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_graph_compile_nodes(m3_session *session,
                                         const m3_graph *graph,
                                         m3_error *error)
{
    size_t index;
    size_t scratch_maximum = 0U;

    for (index = 0U; index < session->node_count; ++index) {
        const m3_graph_node *source = &graph->nodes[index];
        m3_session_node *node = &session->nodes[index];
        m3_status status;

        node->type = source->type;
        if (source->type == M3_GRAPH_NODE_COREML) {
            if (!session->options.allow_neural_engine) {
                return m3_error_set(
                    error, M3_STATUS_UNSUPPORTED,
                    "Core ML partition requires Neural Engine provider");
            }
            node->provider = M3_PROVIDER_NEURAL_ENGINE;
            status = m3_coreml_partition_create(
                &source->payload.coreml, session->views,
                session->value_count, &node->payload.coreml, error);
        } else {
            m3_command *command = &session->commands[index];
            size_t scratch_bytes = 0U;

            node->payload.regular_index = index;
            status = m3_graph_bind_command(
                &source->payload.regular, session->views,
                session->value_count, command, error);
            if (status == M3_STATUS_OK) {
                status = m3_provider_select_regular(
                    session->backend, session->options.allow_metal,
                    session->options.allow_cpu, command, 1U,
                    &node->provider, error);
            }
            if (status == M3_STATUS_OK) {
                status = m3_commands_scratch_bytes(
                    session->backend, command, 1U,
                    &scratch_bytes, error);
            }
            if (scratch_bytes > scratch_maximum) {
                scratch_maximum = scratch_bytes;
            }
        }
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    if (scratch_maximum != 0U) {
        int allocation_status = posix_memalign(
            &session->scratch, M3_GRAPH_STORAGE_ALIGNMENT, scratch_maximum);

        if (allocation_status != 0) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate graph scratch");
        }
        session->scratch_capacity = scratch_maximum;
    }
    return M3_STATUS_OK;
}

m3_status m3_session_create(m3_session **session_output,
                             const m3_graph *graph,
                             const m3_session_options *options,
                             m3_error *error)
{
    m3_session_options selected;
    m3_session *session;
    size_t *producers = NULL;
    size_t *last_uses = NULL;
    size_t index;
    m3_status status;

    if (session_output == NULL || *session_output != NULL || graph == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "session output must be empty and graph is required");
    }
    m3_session_options_init(&selected);
    if (options != NULL) {
        selected = *options;
    }
    if (!selected.allow_neural_engine && !selected.allow_metal &&
        !selected.allow_cpu) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "session has no enabled provider");
    }
    session = calloc(1U, sizeof(*session));
    if (session == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate session");
    }
    session->options = selected;
    session->value_count = graph->value_count;
    session->node_count = graph->node_count;
    session->values = calloc(session->value_count, sizeof(*session->values));
    session->views = calloc(session->value_count, sizeof(*session->views));
    session->slots = calloc(session->value_count, sizeof(*session->slots));
    session->nodes = calloc(session->node_count, sizeof(*session->nodes));
    session->commands = calloc(session->node_count,
                               sizeof(*session->commands));
    producers = calloc(session->value_count, sizeof(*producers));
    last_uses = calloc(session->value_count, sizeof(*last_uses));
    if ((session->value_count != 0U &&
         (session->values == NULL || session->views == NULL ||
          session->slots == NULL || producers == NULL ||
          last_uses == NULL)) ||
        (session->node_count != 0U &&
         (session->nodes == NULL || session->commands == NULL))) {
        status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                              "cannot allocate session plan arrays");
    } else {
        status = M3_STATUS_OK;
    }
    for (index = 0U; status == M3_STATUS_OK &&
                       index < session->value_count; ++index) {
        session->values[index].description =
            graph->values[index].description;
        session->values[index].metadata = graph->values[index].metadata;
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_topology(graph, producers, last_uses, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_backend_create_preferred(
            selected.allow_metal || selected.allow_neural_engine,
            selected.allow_cpu || selected.allow_neural_engine,
            &session->backend, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_plan_slots(session, producers, last_uses, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_allocate_slots(session, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_build_views(session, graph, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_graph_compile_nodes(session, graph, error);
    }
    free(producers);
    free(last_uses);
    if (status != M3_STATUS_OK) {
        m3_session_free(session);
        return status;
    }
    *session_output = session;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
