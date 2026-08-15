/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_weight_stage_internal.h"

#include <stdlib.h>
#include <string.h>

void m3_weight_stage_init(m3_weight_stage *stage)
{
    if (stage != NULL) {
        (void)memset(stage, 0, sizeof(*stage));
    }
}

void m3_weight_stage_dispose(m3_weight_stage *stage)
{
    size_t index;

    if (stage == NULL) {
        return;
    }
    if (stage->storages != NULL) {
        for (index = 0U; index < stage->storage_count; ++index) {
            size_t later;

            for (later = index + 1U; later < stage->storage_count;
                 ++later) {
                if (stage->storages[later] == stage->storages[index]) {
                    stage->storages[later] = NULL;
                }
            }
            m3_storage_free(stage->storages[index]);
        }
    }
    free(stage->storages);
    free(stage->views);
    m3_weight_stage_init(stage);
}

static bool m3_weight_stage_metadata_matches(
    const m3_tensor_metadata *left, const m3_tensor_metadata *right)
{
    uint8_t axis;

    if (left->dtype != right->dtype || left->rank != right->rank ||
        left->element_count != right->element_count ||
        left->byte_count != right->byte_count) {
        return false;
    }
    for (axis = 0U; axis < left->rank; ++axis) {
        if (left->shape[axis] != right->shape[axis]) {
            return false;
        }
    }
    return true;
}

m3_status m3_weight_stage_validate(
    const m3_weight_stage *stage, const m3_weight_table *table,
    m3_backend *backend, m3_error *error)
{
    size_t index;

    if (stage == NULL || table == NULL || backend == NULL ||
        stage->table != table || stage->backend != backend ||
        stage->loaded_bytes != table->aggregate_payload_bytes ||
        stage->storage_count != table->shard_count ||
        stage->view_count != table->binding_count ||
        (table->shard_count != 0U && table->shards == NULL) ||
        (table->binding_count != 0U && table->bindings == NULL) ||
        (stage->storage_count != 0U && stage->storages == NULL) ||
        (stage->view_count != 0U && stage->views == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "published weight stage is inconsistent");
    }
    for (index = 0U; index < table->shard_count; ++index) {
        size_t earlier;

        if (stage->storages[index] == NULL ||
            (uint64_t)m3_storage_size(stage->storages[index]) !=
                table->shards[index].payload_bytes ||
            m3_storage_backend(stage->storages[index]) != backend) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "published weight shard is invalid");
        }
        for (earlier = 0U; earlier < index; ++earlier) {
            if (stage->storages[index] == stage->storages[earlier]) {
                return m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "published weight shards share owned storage");
            }
        }
    }
    for (index = 0U; index < table->binding_count; ++index) {
        const m3_weight_binding *binding = &table->bindings[index];
        const m3_tensor_view *view = &stage->views[index];
        m3_tensor_view checked;
        size_t offset = (size_t)binding->data_start;
        m3_status status;

        if (binding->shard_index >= stage->storage_count ||
            binding->data_end < binding->data_start ||
            binding->data_end - binding->data_start !=
                binding->tensor.byte_count ||
            binding->data_end >
                table->shards[binding->shard_index].payload_bytes ||
            (uint64_t)offset != binding->data_start ||
            view->storage != stage->storages[binding->shard_index] ||
            view->byte_offset != offset ||
            !m3_weight_stage_metadata_matches(
                &view->metadata, &binding->tensor) ||
            !m3_tensor_is_contiguous(view)) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "published weight view is invalid");
        }
        m3_tensor_view_init(&checked);
        status = m3_tensor_reshape(
            view, view->metadata.rank, view->metadata.shape, &checked,
            error);
        if (status != M3_STATUS_OK) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "published weight view is out of bounds");
        }
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_weight_stage_load(m3_weight_stage *stage,
                               const m3_weight_table *table,
                               m3_backend *backend,
                               m3_progress_callback progress,
                               void *progress_context, m3_error *error)
{
    m3_weight_stage_io io;

    m3_weight_stage_io_init(&io);
    return m3_weight_stage_load_with_io(stage, table, backend, progress,
                                        progress_context, &io, error);
}

const m3_tensor_view *m3_weight_stage_find_view(
    const m3_weight_stage *stage, const char *name)
{
    const m3_weight_binding *binding;
    size_t index;

    if (stage == NULL || name == NULL || stage->table == NULL ||
        stage->view_count != stage->table->binding_count ||
        (stage->view_count != 0U && stage->views == NULL)) {
        return NULL;
    }
    binding = m3_weight_table_find(stage->table, name);
    if (binding == NULL) {
        return NULL;
    }
    index = (size_t)(binding - stage->table->bindings);
    return index < stage->view_count ? &stage->views[index] : NULL;
}
