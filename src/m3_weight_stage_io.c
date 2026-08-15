/* SPDX-License-Identifier: GPL-2.0-only */

#if !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include "m3_weight_stage_internal.h"

#include "m3_file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int m3_weight_stage_open_default(void *context, const char *path,
                                        int flags)
{
    int descriptor;
    int saved_errno;

    (void)context;
    descriptor = open(path, flags);
    if (descriptor < 0 || fcntl(descriptor, F_NOCACHE, 1) == 0) {
        return descriptor;
    }
    saved_errno = errno;
    (void)close(descriptor);
    errno = saved_errno;
    return -1;
}

static ssize_t m3_weight_stage_pread_default(void *context, int descriptor,
                                              void *buffer,
                                              size_t byte_count,
                                              off_t byte_offset)
{
    (void)context;
    return pread(descriptor, buffer, byte_count, byte_offset);
}

static int m3_weight_stage_close_default(void *context, int descriptor)
{
    (void)context;
    return close(descriptor);
}

static m3_status m3_weight_stage_write_default(
    void *context, m3_storage *storage, size_t byte_offset,
    const void *source, size_t byte_count, m3_error *error)
{
    (void)context;
    return m3_storage_write(storage, byte_offset, source, byte_count, error);
}

void m3_weight_stage_io_init(m3_weight_stage_io *io)
{
    if (io == NULL) {
        return;
    }
    io->context = NULL;
    io->open_file = m3_weight_stage_open_default;
    io->pread_file = m3_weight_stage_pread_default;
    io->close_file = m3_weight_stage_close_default;
    io->write_storage = m3_weight_stage_write_default;
    io->maximum_chunk_bytes = M3_WEIGHT_STAGE_MAXIMUM_CHUNK_BYTES;
}

m3_status m3_weight_stage_open_shards(const m3_weight_table *table,
                                       const m3_weight_stage_io *io,
                                       int *descriptors, m3_error *error)
{
    size_t index;

    for (index = 0U; index < table->shard_count; ++index) {
        const m3_weight_shard_record *shard = &table->shards[index];
        m3_file_snapshot actual;
        m3_status status;
        int flags = O_RDONLY | O_NONBLOCK;
        int descriptor;

#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif
        descriptor = io->open_file(io->context, shard->path, flags);
        if (descriptor < 0) {
            return m3_error_set(error, M3_STATUS_IO,
                                "cannot open weight shard '%s': %s",
                                shard->path, strerror(errno));
        }
        descriptors[index] = descriptor;
        status = m3_file_snapshot_descriptor(descriptor, &actual, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (!actual.regular_file ||
            !m3_file_snapshot_equal(&actual, &shard->snapshot)) {
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "weight shard '%s' no longer matches its inspected file",
                shard->path);
        }
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static bool m3_weight_stage_off_t(uint64_t value, off_t *converted)
{
    *converted = (off_t)value;
    return *converted >= (off_t)0 && (uint64_t)*converted == value;
}

static m3_status m3_weight_stage_read_exact(
    const m3_weight_stage_io *io, int descriptor, uint8_t *buffer,
    size_t byte_count, uint64_t absolute_offset, const char *path,
    m3_error *error)
{
    size_t filled = 0U;

    while (filled < byte_count) {
        uint64_t current_offset;
        off_t file_offset;
        size_t requested = byte_count - filled;
        ssize_t count;

        if ((uint64_t)filled > UINT64_MAX - absolute_offset) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "weight shard read offset overflows");
        }
        current_offset = absolute_offset + (uint64_t)filled;
        if (!m3_weight_stage_off_t(current_offset, &file_offset)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "weight shard offset does not fit off_t");
        }
        count = io->pread_file(io->context, descriptor, buffer + filled,
                               requested, file_offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count < 0) {
            return m3_error_set(error, M3_STATUS_IO,
                                "cannot read weight shard '%s': %s", path,
                                strerror(errno));
        }
        if (count == 0) {
            return m3_error_set(error, M3_STATUS_IO,
                                "weight shard '%s' ended before its payload",
                                path);
        }
        if ((size_t)count > requested) {
            return m3_error_set(error, M3_STATUS_INTERNAL,
                                "weight shard reader exceeded its request");
        }
        filled += (size_t)count;
    }
    return M3_STATUS_OK;
}

static size_t m3_weight_stage_buffer_size(const m3_weight_stage *stage,
                                           size_t maximum_chunk_bytes)
{
    size_t largest = 0U;
    size_t index;

    for (index = 0U; index < stage->storage_count; ++index) {
        size_t current = m3_storage_size(stage->storages[index]);

        if (current > largest) {
            largest = current;
        }
    }
    return largest < maximum_chunk_bytes ? largest : maximum_chunk_bytes;
}

m3_status m3_weight_stage_read_shards(
    m3_weight_stage *stage, const m3_weight_stage_io *io,
    const int *descriptors, m3_progress_callback progress,
    void *progress_context, m3_error *error)
{
    size_t buffer_size =
        m3_weight_stage_buffer_size(stage, io->maximum_chunk_bytes);
    uint8_t *buffer = NULL;
    uint64_t completed = 0U;
    size_t shard_index;
    m3_status status = M3_STATUS_OK;

    if (buffer_size != 0U) {
        buffer = malloc(buffer_size);
        if (buffer == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate staged weight read buffer");
        }
    }
    for (shard_index = 0U;
         shard_index < stage->storage_count && status == M3_STATUS_OK;
         ++shard_index) {
        const m3_weight_shard_record *shard =
            &stage->table->shards[shard_index];
        uint64_t shard_offset = 0U;

        while (shard_offset < shard->payload_bytes) {
            uint64_t remaining = shard->payload_bytes - shard_offset;
            size_t chunk = remaining < (uint64_t)io->maximum_chunk_bytes
                               ? (size_t)remaining
                               : io->maximum_chunk_bytes;
            uint64_t absolute_offset;
            size_t storage_offset = (size_t)shard_offset;

            if ((uint64_t)storage_offset != shard_offset ||
                shard_offset > UINT64_MAX - shard->data_section_offset) {
                status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                      "weight shard payload offset overflows");
                break;
            }
            absolute_offset = shard->data_section_offset + shard_offset;
            status = m3_weight_stage_read_exact(
                io, descriptors[shard_index], buffer, chunk, absolute_offset,
                shard->path, error);
            if (status == M3_STATUS_OK) {
                status = io->write_storage(
                    io->context, stage->storages[shard_index],
                    storage_offset, buffer, chunk, error);
            }
            if (status != M3_STATUS_OK) {
                break;
            }
            if ((uint64_t)chunk > UINT64_MAX - completed) {
                status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                      "staged weight progress overflows");
                break;
            }
            shard_offset += (uint64_t)chunk;
            completed += (uint64_t)chunk;
            if (progress != NULL &&
                !progress(progress_context, completed,
                          stage->loaded_bytes)) {
                status = m3_error_set(error, M3_STATUS_CANCELLED,
                                      "staged weight load was cancelled");
                break;
            }
        }
    }
    free(buffer);
    return status;
}

m3_status m3_weight_stage_verify_shards(const m3_weight_table *table,
                                         const int *descriptors,
                                         m3_error *error)
{
    size_t index;

    for (index = 0U; index < table->shard_count; ++index) {
        m3_file_snapshot after;
        m3_status status = m3_file_snapshot_descriptor(
            descriptors[index], &after, error);

        if (status != M3_STATUS_OK) {
            return status;
        }
        if (!m3_file_snapshot_equal(&after, &table->shards[index].snapshot)) {
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "weight shard '%s' changed while its payload was read",
                table->shards[index].path);
        }
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_weight_stage_close_shards(const m3_weight_table *table,
                                        const m3_weight_stage_io *io,
                                        int *descriptors, bool report_errors,
                                        m3_error *error)
{
    m3_status status = M3_STATUS_OK;
    size_t index;

    for (index = 0U; index < table->shard_count; ++index) {
        int descriptor = descriptors[index];

        if (descriptor < 0) {
            continue;
        }
        descriptors[index] = -1;
        if (io->close_file(io->context, descriptor) != 0 && report_errors &&
            status == M3_STATUS_OK) {
            status = m3_error_set(error, M3_STATUS_IO,
                                  "cannot close weight shard '%s': %s",
                                  table->shards[index].path,
                                  strerror(errno));
        }
    }
    if (status == M3_STATUS_OK && report_errors) {
        m3_error_reset(error);
    }
    return status;
}
