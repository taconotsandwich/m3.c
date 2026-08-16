/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_graph_internal.h"

#include "m3_provider.h"

#include <stdlib.h>

void m3_session_free(m3_session *session)
{
    size_t index;

    if (session == NULL) {
        return;
    }
    for (index = 0U; index < session->node_count; ++index) {
        if (session->nodes != NULL &&
            session->nodes[index].type == M3_GRAPH_NODE_COREML) {
            m3_coreml_partition_free(
                session->nodes[index].payload.coreml);
        }
    }
    for (index = 0U; index < session->slot_count; ++index) {
        if (session->slots != NULL) {
            m3_storage_free(session->slots[index].storage);
        }
    }
    free(session->scratch);
    free(session->commands);
    free(session->nodes);
    free(session->slots);
    free(session->views);
    free(session->values);
    m3_backend_free(session->backend);
    free(session);
}

static m3_status m3_session_value_check(
    const m3_session *session, m3_graph_value_id value,
    m3_graph_value_role expected, const char *operation,
    m3_error *error)
{
    if (session == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s session is null", operation);
    }
    if ((size_t)value >= session->value_count) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "%s value is out of range", operation);
    }
    if (session->values[value].description.role != expected) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s value has the wrong role", operation);
    }
    return M3_STATUS_OK;
}

m3_status m3_session_write_input(m3_session *session,
                                  m3_graph_value_id value,
                                  const void *data, size_t byte_count,
                                  m3_error *error)
{
    m3_status status = m3_session_value_check(
        session, value, M3_GRAPH_INPUT, "input write", error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (byte_count != session->values[value].metadata.byte_count ||
        (byte_count != 0U && data == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "input write size does not match tensor");
    }
    return m3_storage_write(session->views[value].storage,
                            session->views[value].byte_offset, data,
                            byte_count, error);
}

m3_status m3_session_read_output(const m3_session *session,
                                  m3_graph_value_id value,
                                  void *data, size_t byte_count,
                                  m3_error *error)
{
    m3_status status = m3_session_value_check(
        session, value, M3_GRAPH_OUTPUT, "output read", error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (byte_count != session->values[value].metadata.byte_count ||
        (byte_count != 0U && data == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "output read size does not match tensor");
    }
    return m3_storage_read(session->views[value].storage,
                           session->views[value].byte_offset, data,
                           byte_count, error);
}

static m3_status m3_session_run_regular(m3_session *session,
                                         size_t begin, size_t end,
                                         m3_error *error)
{
    m3_scratch_arena scratch;
    m3_status status = m3_scratch_arena_init(
        &scratch, session->scratch, session->scratch_capacity, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    return m3_provider_execute_regular(
        session->backend, session->nodes[begin].provider,
        &session->commands[begin], end - begin, &scratch, error);
}

m3_status m3_session_run(m3_session *session, m3_error *error)
{
    size_t index = 0U;

    if (session == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "session is null");
    }
    while (index < session->node_count) {
        m3_session_node *node = &session->nodes[index];
        m3_status status;

        if (node->type == M3_GRAPH_NODE_COREML) {
            status = m3_coreml_partition_execute(
                node->payload.coreml, session->views,
                session->value_count, error);
            ++index;
        } else {
            size_t end = index + 1U;

            while (end < session->node_count &&
                   session->nodes[end].type == M3_GRAPH_NODE_REGULAR &&
                   session->nodes[end].provider == node->provider) {
                ++end;
            }
            status = m3_session_run_regular(session, index, end, error);
            index = end;
        }
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_session_get_plan_info(const m3_session *session,
                                    m3_session_plan_info *info,
                                    m3_error *error)
{
    if (session == NULL || info == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "session and plan info output are required");
    }
    info->value_count = session->value_count;
    info->node_count = session->node_count;
    info->storage_slot_count = session->slot_count;
    info->allocated_bytes = session->allocated_bytes;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_session_get_node_provider(const m3_session *session,
                                        size_t node_index,
                                        m3_provider_kind *provider,
                                        m3_error *error)
{
    if (session == NULL || provider == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "session and provider output are required");
    }
    if (node_index >= session->node_count) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "session node index is out of range");
    }
    *provider = session->nodes[node_index].provider;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
