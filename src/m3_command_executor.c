/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_runtime_workspace.h"

#include "m3_provider.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define M3_COMMAND_EXECUTOR_ALIGNMENT 64U

void m3_command_executor_init(m3_command_executor *executor,
                              m3_backend *backend)
{
    if (executor != NULL) {
        (void)memset(executor, 0, sizeof(*executor));
        executor->backend = backend;
    }
}

void m3_command_executor_dispose(m3_command_executor *executor)
{
    if (executor == NULL) {
        return;
    }
    free(executor->scratch);
    m3_command_executor_init(executor, NULL);
}

static m3_status m3_command_executor_grow(m3_command_executor *executor,
                                          size_t required,
                                          m3_error *error)
{
    void *replacement = NULL;
    int allocation_status;

    if (required <= executor->scratch_capacity) {
        return M3_STATUS_OK;
    }
    allocation_status = posix_memalign(&replacement,
                                       M3_COMMAND_EXECUTOR_ALIGNMENT,
                                       required);
    if (allocation_status != 0) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot grow command scratch to %zu bytes",
                            required);
    }
    free(executor->scratch);
    executor->scratch = replacement;
    executor->scratch_capacity = required;
    return M3_STATUS_OK;
}

m3_status m3_command_executor_execute(m3_command_executor *executor,
                                      const m3_command *commands,
                                      size_t command_count,
                                      m3_error *error)
{
    m3_scratch_arena arena;
    m3_provider_kind provider;
    size_t required = 0U;
    m3_status status;

    if (executor == NULL || executor->backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "command executor backend is required");
    }
    status = m3_commands_scratch_bytes(executor->backend, commands,
                                       command_count, &required, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    status = m3_command_executor_grow(executor, required, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    status = m3_scratch_arena_init(&arena, executor->scratch,
                                   executor->scratch_capacity, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    status = m3_provider_select_regular(executor->backend, true, true,
                                        commands, command_count, &provider,
                                        error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    return m3_provider_execute_regular(executor->backend, provider,
                                       commands, command_count, &arena,
                                       error);
}
