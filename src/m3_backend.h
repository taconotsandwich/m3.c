/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_BACKEND_H
#define M3_BACKEND_H

#include "m3_error.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_BACKEND_NAME_CAPACITY 128U

typedef struct m3_backend m3_backend;
typedef struct m3_storage m3_storage;

typedef enum {
    M3_BACKEND_HOST = 0,
    M3_BACKEND_METAL
} m3_backend_kind;

typedef struct {
    char name[M3_BACKEND_NAME_CAPACITY];
    m3_backend_kind kind;
    bool unified_memory;
    uint64_t recommended_working_set_bytes;
    uint64_t maximum_storage_bytes;
} m3_backend_info;

typedef struct {
    size_t live_allocated_bytes;
    size_t peak_allocated_bytes;
    size_t live_storage_count;
    size_t peak_storage_count;
} m3_backend_allocation_stats;

m3_status m3_backend_create_host(m3_backend **backend, m3_error *error);
m3_status m3_backend_create_metal(m3_backend **backend, m3_error *error);
void m3_backend_free(m3_backend *backend);

m3_status m3_backend_get_info(const m3_backend *backend,
                              m3_backend_info *info, m3_error *error);
m3_status m3_backend_get_allocation_stats(
    const m3_backend *backend, m3_backend_allocation_stats *stats,
    m3_error *error);

m3_status m3_storage_allocate(m3_backend *backend, size_t byte_count,
                              size_t alignment, m3_storage **storage,
                              m3_error *error);
void m3_storage_free(m3_storage *storage);
size_t m3_storage_size(const m3_storage *storage);
void *m3_storage_data(m3_storage *storage);
const void *m3_storage_const_data(const m3_storage *storage);
m3_backend *m3_storage_backend(const m3_storage *storage);
m3_status m3_storage_write(m3_storage *storage, size_t byte_offset,
                           const void *source, size_t byte_count,
                           m3_error *error);
m3_status m3_storage_read(const m3_storage *storage, size_t byte_offset,
                          void *destination, size_t byte_count,
                          m3_error *error);

#endif
