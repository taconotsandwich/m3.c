/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_weight_stage_internal.h"

#include <stdlib.h>
#include <string.h>

static m3_status m3_weight_stage_validate_io(const m3_weight_stage_io *io,
                                              m3_error *error)
{
    if (io == NULL || io->open_file == NULL || io->pread_file == NULL ||
        io->close_file == NULL || io->write_storage == NULL ||
        io->maximum_chunk_bytes == 0U ||
        io->maximum_chunk_bytes > M3_WEIGHT_STAGE_MAXIMUM_CHUNK_BYTES) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "staged weight I/O contract is invalid");
    }
    return M3_STATUS_OK;
}

static bool m3_weight_stage_metadata_equal(
    const m3_tensor_metadata *left, const m3_tensor_metadata *right)
{
    uint8_t index;

    if (left->dtype != right->dtype || left->rank != right->rank ||
        left->element_count != right->element_count ||
        left->byte_count != right->byte_count) {
        return false;
    }
    for (index = 0U; index < left->rank; ++index) {
        if (left->shape[index] != right->shape[index]) {
            return false;
        }
    }
    return true;
}

static m3_status m3_weight_stage_validate_shards(
    const m3_weight_table *table, uint64_t *payload_total, m3_error *error)
{
    size_t index;

    *payload_total = 0U;
    if (table->shard_count == 0U || table->binding_count == 0U ||
        table->aggregate_payload_bytes == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "staged component weight table is empty");
    }
    for (index = 0U; index < table->shard_count; ++index) {
        const m3_weight_shard_record *shard = &table->shards[index];
        size_t other;

        if (shard->path == NULL || shard->path[0] == '\0') {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "staged weight shard path is empty");
        }
        if (index != 0U &&
            strcmp(table->shards[index - 1U].path, shard->path) >= 0) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "staged weight shard paths are not unique and sorted");
        }
        if (!shard->snapshot.regular_file || shard->payload_bytes == 0U ||
            shard->data_section_offset > shard->snapshot.file_size ||
            shard->snapshot.file_size - shard->data_section_offset !=
                shard->payload_bytes ||
            shard->snapshot.modification_time_nanoseconds > 999999999U ||
            shard->snapshot.change_time_nanoseconds > 999999999U) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "staged weight shard provenance is inconsistent");
        }
        if (shard->payload_bytes > (uint64_t)SIZE_MAX) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "weight shard payload does not fit size_t");
        }
        if (shard->payload_bytes > UINT64_MAX - *payload_total) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "staged weight payload total overflows");
        }
        *payload_total += shard->payload_bytes;
        for (other = 0U; other < index; ++other) {
            if (table->shards[other].snapshot.device ==
                    shard->snapshot.device &&
                table->shards[other].snapshot.inode ==
                    shard->snapshot.inode) {
                return m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "staged weight shard paths alias the same file");
            }
        }
    }
    if (*payload_total != table->aggregate_payload_bytes) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "staged weight payload aggregate is inconsistent");
    }
    return M3_STATUS_OK;
}

static m3_status m3_weight_stage_validate_bindings(
    const m3_weight_table *table, m3_error *error)
{
    size_t index;

    for (index = 0U; index < table->binding_count; ++index) {
        const m3_weight_binding *binding = &table->bindings[index];
        const m3_weight_shard_record *shard;
        m3_tensor_metadata checked;
        uint64_t range_bytes;
        m3_status status;

        if (binding->name == NULL || binding->name[0] == '\0') {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "staged weight binding name is empty");
        }
        if (index != 0U &&
            strcmp(table->bindings[index - 1U].name, binding->name) >= 0) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "staged weight bindings are not unique and sorted");
        }
        if (binding->shard_index >= table->shard_count) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "staged weight binding shard is out of range");
        }
        shard = &table->shards[binding->shard_index];
        if (binding->data_end < binding->data_start ||
            binding->data_end > shard->payload_bytes) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "staged weight binding range is invalid");
        }
        range_bytes = binding->data_end - binding->data_start;
        if (range_bytes != (uint64_t)binding->tensor.byte_count) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "staged weight binding byte count is inconsistent");
        }
        if (binding->tensor.dtype != M3_DTYPE_F32 &&
            binding->tensor.dtype != M3_DTYPE_F16 &&
            binding->tensor.dtype != M3_DTYPE_BF16) {
            return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                                "staged weight binding dtype is unsupported");
        }
        status = m3_tensor_metadata_init(&checked, binding->tensor.dtype,
                                         binding->tensor.rank,
                                         binding->tensor.shape, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (!m3_weight_stage_metadata_equal(&checked, &binding->tensor)) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "staged weight binding metadata is inconsistent");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_weight_stage_preflight(
    const m3_weight_table *table, m3_backend *backend, m3_error *error)
{
    m3_backend_allocation_stats stats;
    m3_backend_info info;
    uint64_t payload_total;
    uint64_t live_bytes;
    size_t aggregate_size;
    size_t index;
    m3_status status;

    if ((table->shard_count != 0U && table->shards == NULL) ||
        (table->binding_count != 0U && table->bindings == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "staged weight table storage is missing");
    }
    if (table->shard_count > SIZE_MAX / sizeof(m3_storage *) ||
        table->shard_count > SIZE_MAX / sizeof(int) ||
        table->binding_count > SIZE_MAX / sizeof(m3_tensor_view)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "staged weight state allocation overflows");
    }
    status = m3_weight_stage_validate_shards(table, &payload_total, error);
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_validate_bindings(table, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    status = m3_backend_get_info(backend, &info, error);
    if (status == M3_STATUS_OK) {
        status = m3_backend_get_allocation_stats(backend, &stats, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < table->shard_count; ++index) {
        if (info.maximum_storage_bytes == 0U ||
            table->shards[index].payload_bytes >
                info.maximum_storage_bytes) {
            return m3_error_set(
                error, M3_STATUS_OUT_OF_RANGE,
                "weight shard exceeds backend storage limit");
        }
    }
    aggregate_size = (size_t)payload_total;
    if ((uint64_t)aggregate_size != payload_total ||
        stats.live_allocated_bytes > SIZE_MAX - aggregate_size) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "staged weight working set size overflows");
    }
    live_bytes = (uint64_t)stats.live_allocated_bytes;
    if ((size_t)live_bytes != stats.live_allocated_bytes ||
        payload_total > UINT64_MAX - live_bytes) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "staged weight working set total overflows");
    }
    if (info.recommended_working_set_bytes != 0U &&
        live_bytes + payload_total > info.recommended_working_set_bytes) {
        return m3_error_set(
            error, M3_STATUS_OUT_OF_MEMORY,
            "staged weights exceed the backend recommended working set");
    }
    return M3_STATUS_OK;
}

static m3_status m3_weight_stage_allocate(m3_weight_stage *built,
                                           m3_error *error)
{
    size_t index;

    for (index = 0U; index < built->storage_count; ++index) {
        size_t byte_count =
            (size_t)built->table->shards[index].payload_bytes;
        m3_status status = m3_storage_allocate(
            built->backend, byte_count, 64U, &built->storages[index], error);

        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_weight_stage_build_views(m3_weight_stage *built,
                                              m3_error *error)
{
    size_t index;

    for (index = 0U; index < built->view_count; ++index) {
        const m3_weight_binding *binding = &built->table->bindings[index];
        uint64_t range_bytes = binding->data_end - binding->data_start;
        size_t byte_offset = (size_t)binding->data_start;
        m3_status status;

        if ((uint64_t)byte_offset != binding->data_start) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "weight view offset does not fit size_t");
        }
        status = m3_tensor_view_contiguous(
            &built->views[index], built->storages[binding->shard_index],
            binding->tensor.dtype, binding->tensor.rank,
            binding->tensor.shape, byte_offset, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (built->views[index].metadata.byte_count != (size_t)range_bytes ||
            !m3_weight_stage_metadata_equal(&built->views[index].metadata,
                                            &binding->tensor)) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "weight view disagrees with its binding");
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_weight_stage_prepare_arrays(m3_weight_stage *built,
                                                 int **descriptors,
                                                 m3_error *error)
{
    size_t index;

    if (built->storage_count != 0U) {
        *descriptors = malloc(built->storage_count * sizeof(**descriptors));
        if (*descriptors == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate staged weight shard state");
        }
        for (index = 0U; index < built->storage_count; ++index) {
            (*descriptors)[index] = -1;
        }
        built->storages = calloc(built->storage_count,
                                 sizeof(*built->storages));
        if (built->storages == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate staged weight shard state");
        }
    }
    if (built->view_count != 0U) {
        built->views = calloc(built->view_count, sizeof(*built->views));
        if (built->views == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate staged weight views");
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_weight_stage_load_with_io(
    m3_weight_stage *stage, const m3_weight_table *table,
    m3_backend *backend, m3_weight_stage_progress progress,
    void *progress_context, const m3_weight_stage_io *io, m3_error *error)
{
    m3_weight_stage built;
    int *descriptors = NULL;
    m3_status status;

    if (stage == NULL || table == NULL || backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "stage, weight table, and backend are required");
    }
    status = m3_weight_stage_validate_io(io, error);
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_preflight(table, backend, error);
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (progress != NULL &&
        !progress(progress_context, 0U, table->aggregate_payload_bytes)) {
        return m3_error_set(error, M3_STATUS_CANCELLED,
                            "staged weight load was cancelled");
    }
    m3_weight_stage_init(&built);
    built.table = table;
    built.backend = backend;
    built.storage_count = table->shard_count;
    built.view_count = table->binding_count;
    built.loaded_bytes = table->aggregate_payload_bytes;
    status = m3_weight_stage_prepare_arrays(&built, &descriptors, error);
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_open_shards(table, io, descriptors, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_allocate(&built, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_build_views(&built, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_read_shards(
            &built, io, descriptors, progress, progress_context, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_verify_shards(table, descriptors, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_weight_stage_close_shards(
            table, io, descriptors, true, error);
    } else if (descriptors != NULL) {
        (void)m3_weight_stage_close_shards(
            table, io, descriptors, false, error);
    }
    free(descriptors);
    if (status != M3_STATUS_OK) {
        m3_weight_stage_dispose(&built);
        return status;
    }
    m3_weight_stage_dispose(stage);
    *stage = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
