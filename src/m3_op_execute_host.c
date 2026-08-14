/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_op_internal.h"

static m3_status m3_host_execute_one(const m3_command *command,
                                     m3_scratch_arena *scratch,
                                     m3_error *error)
{
    bool handled = false;
    m3_status status = m3_command_data_preflight(command, error);

    if (status == M3_STATUS_OK) {
        status = m3_host_execute_basic(command, scratch, &handled, error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_host_execute_dense(command, scratch, &handled, error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_host_execute_attention(command, scratch, &handled,
                                           error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_host_execute_sampling(command, scratch, &handled,
                                          error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_host_execute_convolution(command, scratch, &handled,
                                             error);
    }
    if (status == M3_STATUS_OK && !handled) {
        status = m3_error_set(error, M3_STATUS_INTERNAL,
                              "validated operation has no host executor");
    }
    return status;
}

m3_status m3_host_execute_commands(void *context,
                                   const m3_command *commands,
                                   size_t command_count,
                                   m3_scratch_arena *scratch,
                                   m3_error *error)
{
    size_t base_mark = m3_scratch_arena_mark(scratch);
    size_t index;

    (void)context;
    for (index = 0U; index < command_count; ++index) {
        m3_status status = m3_host_execute_one(&commands[index], scratch,
                                               error);

        if (scratch != NULL) {
            (void)m3_scratch_arena_rewind(scratch, base_mark, NULL);
        }
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}
