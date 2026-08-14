/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

NSString *m3_metal_source_attention(void)
{
    return @"";
}

m3_status m3_metal_prepare_attention(m3_metal_context *context,
                                      m3_error *error)
{
    (void)context;
    (void)error;
    return M3_STATUS_OK;
}

bool m3_metal_attention_writes_storage(const m3_command *command,
                                        const m3_storage *storage)
{
    (void)command;
    (void)storage;
    return false;
}

m3_status m3_metal_preflight_attention(const m3_command *commands,
                                        size_t command_index,
                                        m3_error *error)
{
    (void)commands;
    (void)command_index;
    (void)error;
    return M3_STATUS_OK;
}

m3_status m3_metal_encode_attention(const m3_metal_context *context,
                                     id<MTLComputeCommandEncoder> encoder,
                                     const m3_command *command,
                                     bool *handled, m3_error *error)
{
    (void)context;
    (void)encoder;
    (void)command;
    (void)error;
    *handled = false;
    return M3_STATUS_OK;
}
