/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_BACKEND_INTERNAL_H
#define M3_BACKEND_INTERNAL_H

#include "m3_backend.h"
#include "m3_op.h"

typedef struct {
    void (*destroy_context)(void *context);
    m3_status (*allocate)(void *context, size_t byte_count,
                          size_t alignment, void **handle, void **data,
                          m3_error *error);
    void (*free_storage)(void *context, void *handle, void *data);
    m3_status (*execute)(void *context, const m3_command *commands,
                         size_t command_count, m3_scratch_arena *scratch,
                         m3_error *error);
} m3_backend_vtable;

m3_status m3_backend_create_internal(const m3_backend_vtable *vtable,
                                     void *context,
                                     const m3_backend_info *info,
                                     m3_backend **backend,
                                     m3_error *error);
void *m3_storage_handle_internal(const m3_storage *storage);

#endif
