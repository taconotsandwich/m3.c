/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_metal_internal.h"

m3_status m3_metal_encode_convolution(
    const m3_metal_context *context,
    id<MTLComputeCommandEncoder> encoder, const m3_command *command,
    bool *handled, m3_error *error)
{
    (void)context;
    (void)encoder;
    (void)command;
    (void)error;
    *handled = false;
    return M3_STATUS_OK;
}
