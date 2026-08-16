/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_provider.h"

#include "m3_op_internal.h"

static bool m3_provider_metal_kind(m3_op_kind kind)
{
    switch (kind) {
    case M3_OP_COPY:
    case M3_OP_CAST:
    case M3_OP_ADD:
    case M3_OP_MUL:
    case M3_OP_EMBEDDING:
    case M3_OP_MATMUL:
    case M3_OP_LINEAR:
    case M3_OP_RMS_NORM:
    case M3_OP_LAYER_NORM:
    case M3_OP_ROPE:
    case M3_OP_ATTENTION:
    case M3_OP_GATED_SILU:
    case M3_OP_SOFTMAX:
    case M3_OP_TOP_K:
    case M3_OP_CATEGORICAL:
    case M3_OP_CONV1D:
    case M3_OP_CONV_TRANSPOSE1D:
    case M3_OP_NEAREST_RESIZE1D:
    case M3_OP_SNAKE1D:
    case M3_OP_TANH:
        return true;
    }
    return false;
}

m3_status m3_backend_create_preferred(bool allow_metal, bool allow_cpu,
                                       m3_backend **backend,
                                       m3_error *error)
{
    m3_status status = M3_STATUS_UNSUPPORTED;

    if (backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "preferred backend output is null");
    }
    *backend = NULL;
    if (!allow_metal && !allow_cpu) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "at least one storage provider is required");
    }
    if (allow_metal) {
        status = m3_backend_create_metal(backend, error);
        if (status == M3_STATUS_OK || status != M3_STATUS_UNSUPPORTED) {
            return status;
        }
    }
    if (allow_cpu) {
        return m3_backend_create_host(backend, error);
    }
    return status;
}

m3_status m3_provider_select_regular(
    const m3_backend *storage_backend, bool allow_metal, bool allow_cpu,
    const m3_command *commands, size_t command_count,
    m3_provider_kind *provider, m3_error *error)
{
    m3_backend_info info;
    size_t index;
    m3_status status;

    if (storage_backend == NULL || provider == NULL ||
        (command_count != 0U && commands == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "regular provider selection is incomplete");
    }
    status = m3_backend_get_info(storage_backend, &info, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (allow_metal && info.kind == M3_BACKEND_METAL) {
        for (index = 0U; index < command_count; ++index) {
            if (!m3_provider_metal_kind(commands[index].kind)) {
                break;
            }
        }
        if (index == command_count) {
            *provider = M3_PROVIDER_METAL;
            m3_error_reset(error);
            return M3_STATUS_OK;
        }
    }
    if (allow_cpu) {
        *provider = M3_PROVIDER_CPU;
        m3_error_reset(error);
        return M3_STATUS_OK;
    }
    return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                        "no enabled provider supports the command list");
}

m3_status m3_provider_execute_regular(
    m3_backend *storage_backend, m3_provider_kind provider,
    const m3_command *commands, size_t command_count,
    m3_scratch_arena *scratch, m3_error *error)
{
    size_t required = 0U;
    m3_status status;

    if (provider == M3_PROVIDER_METAL) {
        return m3_backend_execute(storage_backend, commands, command_count,
                                  scratch, error);
    }
    if (provider != M3_PROVIDER_CPU) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "provider cannot execute regular operations");
    }
    status = m3_commands_scratch_bytes(storage_backend, commands,
                                       command_count, &required, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (scratch == NULL || scratch->offset > scratch->capacity ||
        required > scratch->capacity - scratch->offset) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "CPU provider needs %zu scratch bytes", required);
    }
    return m3_host_execute_commands(NULL, commands, command_count, scratch,
                                    error);
}
