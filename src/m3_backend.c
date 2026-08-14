/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_backend_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct m3_storage {
    m3_backend *owner;
    void *handle;
    void *data;
    size_t byte_count;
    struct m3_storage *previous;
    struct m3_storage *next;
};

struct m3_backend {
    const m3_backend_vtable *vtable;
    void *context;
    m3_backend_info info;
    m3_backend_allocation_stats stats;
    m3_storage *storages;
};

static bool m3_backend_power_of_two(size_t value)
{
    return value != 0U && (value & (value - 1U)) == 0U;
}

static m3_status m3_backend_validate_range(size_t storage_size,
                                           size_t byte_offset,
                                           size_t byte_count,
                                           const void *memory,
                                           const char *operation,
                                           m3_error *error)
{
    if (byte_offset > storage_size ||
        byte_count > storage_size - byte_offset) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "%s range exceeds storage", operation);
    }
    if (byte_count != 0U && memory == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s memory is null", operation);
    }
    return M3_STATUS_OK;
}

static void m3_backend_link_storage(m3_backend *backend,
                                    m3_storage *storage)
{
    storage->owner = backend;
    storage->next = backend->storages;
    if (backend->storages != NULL) {
        backend->storages->previous = storage;
    }
    backend->storages = storage;
}

static void m3_backend_unlink_storage(m3_storage *storage)
{
    m3_backend *backend = storage->owner;

    if (storage->previous != NULL) {
        storage->previous->next = storage->next;
    } else {
        backend->storages = storage->next;
    }
    if (storage->next != NULL) {
        storage->next->previous = storage->previous;
    }
    backend->stats.live_allocated_bytes -= storage->byte_count;
    --backend->stats.live_storage_count;
    storage->owner = NULL;
    storage->previous = NULL;
    storage->next = NULL;
}

m3_status m3_backend_create_internal(const m3_backend_vtable *vtable,
                                     void *context,
                                     const m3_backend_info *info,
                                     m3_backend **backend_output,
                                     m3_error *error)
{
    m3_backend *backend;

    if (backend_output == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "backend output is null");
    }
    *backend_output = NULL;
    if (vtable == NULL || vtable->destroy_context == NULL ||
        vtable->allocate == NULL || vtable->free_storage == NULL ||
        context == NULL || info == NULL || info->name[0] == '\0') {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "backend lifetime contract is incomplete");
    }
    backend = calloc(1U, sizeof(*backend));
    if (backend == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate backend state");
    }
    backend->vtable = vtable;
    backend->context = context;
    backend->info = *info;
    *backend_output = backend;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

void m3_backend_free(m3_backend *backend)
{
    m3_storage *storage;

    if (backend == NULL) {
        return;
    }
    storage = backend->storages;
    backend->storages = NULL;
    while (storage != NULL) {
        m3_storage *next = storage->next;

        backend->vtable->free_storage(backend->context, storage->handle,
                                      storage->data);
        free(storage);
        storage = next;
    }
    backend->vtable->destroy_context(backend->context);
    free(backend);
}

m3_status m3_backend_get_info(const m3_backend *backend,
                              m3_backend_info *info, m3_error *error)
{
    if (backend == NULL || info == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "backend and info output are required");
    }
    *info = backend->info;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_backend_get_allocation_stats(
    const m3_backend *backend, m3_backend_allocation_stats *stats,
    m3_error *error)
{
    if (backend == NULL || stats == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "backend and stats output are required");
    }
    *stats = backend->stats;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_storage_allocate(m3_backend *backend, size_t byte_count,
                              size_t alignment, m3_storage **storage_output,
                              m3_error *error)
{
    m3_storage *storage;
    void *handle = NULL;
    void *data = NULL;
    m3_status status;

    if (storage_output == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "storage output is null");
    }
    *storage_output = NULL;
    if (backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "storage backend is null");
    }
    if (!m3_backend_power_of_two(alignment) ||
        alignment < sizeof(void *)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "storage alignment is invalid");
    }
    if ((uint64_t)byte_count > backend->info.maximum_storage_bytes) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "storage exceeds backend allocation limit");
    }
    if (backend->stats.live_allocated_bytes > SIZE_MAX - byte_count ||
        backend->stats.live_storage_count == SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "backend allocation statistics overflow");
    }
    storage = calloc(1U, sizeof(*storage));
    if (storage == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate storage state");
    }
    status = backend->vtable->allocate(backend->context, byte_count,
                                       alignment, &handle, &data, error);
    if (status != M3_STATUS_OK) {
        free(storage);
        return status;
    }
    if (byte_count != 0U && data == NULL) {
        backend->vtable->free_storage(backend->context, handle, data);
        free(storage);
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "backend returned null storage data");
    }
    if (data != NULL && (uintptr_t)data % alignment != 0U) {
        backend->vtable->free_storage(backend->context, handle, data);
        free(storage);
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "backend returned misaligned storage data");
    }
    storage->handle = handle;
    storage->data = data;
    storage->byte_count = byte_count;
    m3_backend_link_storage(backend, storage);
    backend->stats.live_allocated_bytes += byte_count;
    ++backend->stats.live_storage_count;
    if (backend->stats.live_allocated_bytes >
        backend->stats.peak_allocated_bytes) {
        backend->stats.peak_allocated_bytes =
            backend->stats.live_allocated_bytes;
    }
    if (backend->stats.live_storage_count >
        backend->stats.peak_storage_count) {
        backend->stats.peak_storage_count =
            backend->stats.live_storage_count;
    }
    *storage_output = storage;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

void m3_storage_free(m3_storage *storage)
{
    m3_backend *backend;

    if (storage == NULL) {
        return;
    }
    backend = storage->owner;
    if (backend != NULL) {
        m3_backend_unlink_storage(storage);
        backend->vtable->free_storage(backend->context, storage->handle,
                                      storage->data);
    }
    free(storage);
}

size_t m3_storage_size(const m3_storage *storage)
{
    return storage == NULL ? 0U : storage->byte_count;
}

void *m3_storage_data(m3_storage *storage)
{
    return storage == NULL ? NULL : storage->data;
}

const void *m3_storage_const_data(const m3_storage *storage)
{
    return storage == NULL ? NULL : storage->data;
}

m3_backend *m3_storage_backend(const m3_storage *storage)
{
    return storage == NULL ? NULL : storage->owner;
}

m3_status m3_storage_write(m3_storage *storage, size_t byte_offset,
                           const void *source, size_t byte_count,
                           m3_error *error)
{
    m3_status status;

    if (storage == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "write storage is null");
    }
    status = m3_backend_validate_range(storage->byte_count, byte_offset,
                                       byte_count, source, "storage write",
                                       error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (byte_count != 0U) {
        (void)memcpy((uint8_t *)storage->data + byte_offset, source,
                     byte_count);
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_storage_read(const m3_storage *storage, size_t byte_offset,
                          void *destination, size_t byte_count,
                          m3_error *error)
{
    m3_status status;

    if (storage == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "read storage is null");
    }
    status = m3_backend_validate_range(storage->byte_count, byte_offset,
                                       byte_count, destination,
                                       "storage read", error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (byte_count != 0U) {
        (void)memcpy(destination,
                     (const uint8_t *)storage->data + byte_offset,
                     byte_count);
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}
