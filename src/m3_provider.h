/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_PROVIDER_H
#define M3_PROVIDER_H

#include "m3_op.h"

m3_status m3_backend_create_preferred(bool allow_metal, bool allow_cpu,
                                       m3_backend **backend,
                                       m3_error *error);
m3_status m3_provider_select_regular(
    const m3_backend *storage_backend, bool allow_metal, bool allow_cpu,
    const m3_command *commands, size_t command_count,
    m3_provider_kind *provider, m3_error *error);
m3_status m3_provider_execute_regular(
    m3_backend *storage_backend, m3_provider_kind provider,
    const m3_command *commands, size_t command_count,
    m3_scratch_arena *scratch, m3_error *error);

#endif
