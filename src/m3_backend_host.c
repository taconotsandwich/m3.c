/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_internal.h"
#include "m3_op_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned int marker;
} m3_host_context;

static void m3_host_destroy(void *context)
{
    free(context);
}

static m3_status m3_host_allocate(void *context, size_t byte_count,
                                  size_t alignment, void **handle,
                                  void **data, m3_error *error)
{
    size_t allocation_size = byte_count;
    size_t remainder;
    void *memory;

    (void)context;
    *handle = NULL;
    *data = NULL;
    if (byte_count == 0U) {
        return M3_STATUS_OK;
    }
    remainder = allocation_size & (alignment - 1U);
    if (remainder != 0U) {
        if (allocation_size > SIZE_MAX - (alignment - remainder)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "host allocation size overflows");
        }
        allocation_size += alignment - remainder;
    }
    memory = aligned_alloc(alignment, allocation_size);
    if (memory == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate %zu host storage bytes",
                            byte_count);
    }
    *handle = memory;
    *data = memory;
    return M3_STATUS_OK;
}

static void m3_host_free(void *context, void *handle, void *data)
{
    (void)context;
    (void)data;
    free(handle);
}

m3_status m3_backend_create_host(m3_backend **backend, m3_error *error)
{
    static const m3_backend_vtable vtable = {
        m3_host_destroy,
        m3_host_allocate,
        m3_host_free,
        m3_host_execute_commands
    };
    m3_backend_info info;
    m3_host_context *context;
    m3_status status;

    if (backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "host backend output is null");
    }
    *backend = NULL;
    context = calloc(1U, sizeof(*context));
    if (context == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate host backend context");
    }
    (void)memset(&info, 0, sizeof(info));
    (void)memcpy(info.name, "Host reference storage",
                 sizeof("Host reference storage"));
    info.kind = M3_BACKEND_HOST;
    info.unified_memory = true;
    info.maximum_storage_bytes = (uint64_t)SIZE_MAX;
    status = m3_backend_create_internal(&vtable, context, &info, backend,
                                        error);
    if (status != M3_STATUS_OK) {
        free(context);
    }
    return status;
}
