/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_weights.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const m3_weight_shard_source *source;
} m3_weight_source_order;

typedef struct {
    uint64_t start;
    uint64_t end;
} m3_weight_range;

static m3_status m3_weight_copy_string(const char *source, char **copy_output,
                                       m3_error *error)
{
    size_t length;
    char *copy;

    if (copy_output == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight string output is null");
    }
    *copy_output = NULL;
    if (source == NULL || source[0] == '\0') {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight name or shard path is empty");
    }
    length = strlen(source);
    if (length == SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "weight string length overflows");
    }
    copy = malloc(length + 1U);
    if (copy == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot copy weight string");
    }
    (void)memcpy(copy, source, length + 1U);
    *copy_output = copy;
    return M3_STATUS_OK;
}

static bool m3_weight_metadata_equal(const m3_tensor_metadata *left,
                                     const m3_tensor_metadata *right)
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

static m3_status m3_weight_validate_metadata(
    const m3_tensor_metadata *metadata, m3_error *error)
{
    m3_tensor_metadata checked;
    m3_status status;

    if (metadata->dtype != M3_DTYPE_F32 && metadata->dtype != M3_DTYPE_F16 &&
        metadata->dtype != M3_DTYPE_BF16) {
        return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                            "weight dtype must be F32, F16, or BF16");
    }
    status = m3_tensor_metadata_init(&checked, metadata->dtype,
                                     metadata->rank, metadata->shape, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (!m3_weight_metadata_equal(&checked, metadata)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "weight metadata counts are inconsistent");
    }
    return M3_STATUS_OK;
}

static int m3_weight_compare_sources(const void *left, const void *right)
{
    const m3_weight_source_order *left_order = left;
    const m3_weight_source_order *right_order = right;

    return strcmp(left_order->source->path, right_order->source->path);
}

static int m3_weight_compare_bindings(const void *left, const void *right)
{
    const m3_weight_binding *left_binding = left;
    const m3_weight_binding *right_binding = right;

    return strcmp(left_binding->name, right_binding->name);
}

static int m3_weight_compare_ranges(const void *left, const void *right)
{
    const m3_weight_range *left_range = left;
    const m3_weight_range *right_range = right;

    if (left_range->start < right_range->start) {
        return -1;
    }
    if (left_range->start > right_range->start) {
        return 1;
    }
    if (left_range->end < right_range->end) {
        return -1;
    }
    return left_range->end > right_range->end ? 1 : 0;
}

void m3_weight_table_init(m3_weight_table *table)
{
    if (table != NULL) {
        (void)memset(table, 0, sizeof(*table));
    }
}

void m3_weight_table_dispose(m3_weight_table *table)
{
    size_t index;

    if (table == NULL) {
        return;
    }
    for (index = 0U; index < table->shard_count; ++index) {
        free(table->shards[index].path);
    }
    for (index = 0U; index < table->binding_count; ++index) {
        free(table->bindings[index].name);
    }
    free(table->shards);
    free(table->bindings);
    m3_weight_table_init(table);
}

static m3_status m3_weight_validate_snapshot(
    const m3_safetensors_metadata *metadata, uint64_t *payload_bytes,
    m3_error *error)
{
    const m3_file_snapshot *snapshot = &metadata->source_snapshot;

    if (!snapshot->regular_file ||
        snapshot->modification_time_nanoseconds > 999999999U ||
        snapshot->change_time_nanoseconds > 999999999U ||
        metadata->data_section_offset > snapshot->file_size) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "weight shard snapshot is inconsistent");
    }
    *payload_bytes = snapshot->file_size - metadata->data_section_offset;
    if (*payload_bytes == 0U || metadata->tensor_count == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "weight shard is empty");
    }
    if (metadata->tensors == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight shard tensor inventory is null");
    }
    return M3_STATUS_OK;
}

static m3_status m3_weight_validate_ranges(
    const m3_safetensors_metadata *metadata, uint64_t payload_bytes,
    m3_error *error)
{
    m3_weight_range *ranges;
    uint64_t tensor_bytes = 0U;
    uint64_t expected_start = 0U;
    size_t index;
    m3_status status = M3_STATUS_OK;

    if (metadata->tensor_count > SIZE_MAX / sizeof(*ranges)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "weight range allocation overflows");
    }
    ranges = malloc(metadata->tensor_count * sizeof(*ranges));
    if (ranges == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate weight range validation");
    }
    for (index = 0U; index < metadata->tensor_count; ++index) {
        const m3_safetensors_tensor *tensor = &metadata->tensors[index];
        uint64_t range_bytes;

        if (tensor->name == NULL || tensor->name[0] == '\0') {
            status = m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                  "inspected weight name is empty");
            break;
        }
        status = m3_weight_validate_metadata(&tensor->tensor, error);
        if (status != M3_STATUS_OK) {
            break;
        }
        if (tensor->data_end < tensor->data_start) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "weight '%s' has a reversed byte range",
                                  tensor->name);
            break;
        }
        if (tensor->data_end > payload_bytes) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "weight '%s' exceeds its shard payload",
                                  tensor->name);
            break;
        }
        range_bytes = tensor->data_end - tensor->data_start;
        if (range_bytes != (uint64_t)tensor->tensor.byte_count) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "weight '%s' range disagrees with metadata",
                                  tensor->name);
            break;
        }
        if (range_bytes > UINT64_MAX - tensor_bytes) {
            status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                  "weight shard byte total overflows");
            break;
        }
        tensor_bytes += range_bytes;
        ranges[index].start = tensor->data_start;
        ranges[index].end = tensor->data_end;
    }
    if (status == M3_STATUS_OK) {
        qsort(ranges, metadata->tensor_count, sizeof(*ranges),
              m3_weight_compare_ranges);
        for (index = 0U; index < metadata->tensor_count; ++index) {
            if (ranges[index].start != expected_start) {
                status = m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "weight shard ranges contain a gap or overlap");
                break;
            }
            expected_start = ranges[index].end;
        }
    }
    if (status == M3_STATUS_OK &&
        (expected_start != payload_bytes || tensor_bytes != payload_bytes ||
         (uint64_t)metadata->tensor_bytes != payload_bytes)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "weight shard payload total is inconsistent");
    }
    free(ranges);
    return status;
}

static m3_status m3_weight_copy_shard(
    m3_weight_shard_record *record, const m3_weight_shard_source *source,
    uint64_t payload_bytes, m3_error *error)
{
    m3_status status;

    if (record == NULL || source == NULL || source->metadata == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "owned weight shard copy argument is null");
    }
    status = m3_weight_copy_string(source->path, &record->path, error);

    if (status == M3_STATUS_OK) {
        record->data_section_offset = source->metadata->data_section_offset;
        record->payload_bytes = payload_bytes;
        record->snapshot = source->metadata->source_snapshot;
    }
    return status;
}

static m3_status m3_weight_copy_binding(
    m3_weight_binding *binding, const m3_safetensors_tensor *tensor,
    size_t shard_index, m3_error *error)
{
    m3_status status;

    if (binding == NULL || tensor == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "owned weight binding copy argument is null");
    }
    status = m3_weight_copy_string(tensor->name, &binding->name, error);

    if (status == M3_STATUS_OK) {
        binding->tensor = tensor->tensor;
        binding->shard_index = shard_index;
        binding->data_start = tensor->data_start;
        binding->data_end = tensor->data_end;
    }
    return status;
}

m3_status m3_weight_table_build(m3_weight_table *table,
                                const m3_weight_shard_source *shards,
                                size_t shard_count, m3_error *error)
{
    m3_weight_source_order *order = NULL;
    m3_weight_table built;
    size_t total_bindings = 0U;
    size_t output_index = 0U;
    size_t shard_index;
    m3_status status = M3_STATUS_OK;

    if (table == NULL || (shard_count != 0U && shards == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight table and shards are required");
    }
    m3_weight_table_init(&built);
    if (shard_count > SIZE_MAX / sizeof(*order) ||
        shard_count > SIZE_MAX / sizeof(*built.shards)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "weight shard allocation overflows");
    }
    if (shard_count != 0U) {
        order = malloc(shard_count * sizeof(*order));
        if (order == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate weight shard order");
        }
    }
    for (shard_index = 0U; shard_index < shard_count; ++shard_index) {
        if (shards[shard_index].path == NULL ||
            shards[shard_index].path[0] == '\0' ||
            shards[shard_index].metadata == NULL) {
            (void)m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                               "weight shard descriptor is incomplete");
            status = M3_STATUS_INVALID_ARGUMENT;
            break;
        }
        order[shard_index].source = &shards[shard_index];
    }
    if (status == M3_STATUS_OK && shard_count > 1U) {
        qsort(order, shard_count, sizeof(*order), m3_weight_compare_sources);
    }
    for (shard_index = 0U;
         shard_index < shard_count && status == M3_STATUS_OK; ++shard_index) {
        const m3_safetensors_metadata *metadata =
            order[shard_index].source->metadata;
        uint64_t payload_bytes = 0U;
        size_t alias_index;

        if (shard_index != 0U &&
            strcmp(order[shard_index - 1U].source->path,
                   order[shard_index].source->path) == 0) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "duplicate weight shard path '%s'",
                                  order[shard_index].source->path);
            break;
        }
        status = m3_weight_validate_snapshot(metadata, &payload_bytes, error);
        if (status != M3_STATUS_OK) {
            break;
        }
        for (alias_index = 0U; alias_index < shard_index; ++alias_index) {
            const m3_file_snapshot *other =
                &order[alias_index].source->metadata->source_snapshot;
            const m3_file_snapshot *current = &metadata->source_snapshot;

            if (current->device == other->device &&
                current->inode == other->inode) {
                status = m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "weight shard paths alias the same file");
                break;
            }
        }
        if (status == M3_STATUS_OK) {
            status = m3_weight_validate_ranges(metadata, payload_bytes,
                                               error);
        }
        if (status == M3_STATUS_OK &&
            metadata->tensor_count > SIZE_MAX - total_bindings) {
            status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                  "weight binding count overflows");
        }
        if (status == M3_STATUS_OK &&
            payload_bytes > UINT64_MAX - built.aggregate_payload_bytes) {
            status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                  "weight payload aggregate overflows");
        }
        if (status == M3_STATUS_OK) {
            total_bindings += metadata->tensor_count;
            built.aggregate_payload_bytes += payload_bytes;
        }
    }
    if (status == M3_STATUS_OK &&
        total_bindings > SIZE_MAX / sizeof(*built.bindings)) {
        status = m3_error_set(error, M3_STATUS_OVERFLOW,
                              "weight binding allocation overflows");
    }
    if (status == M3_STATUS_OK && shard_count != 0U) {
        built.shards = calloc(shard_count, sizeof(*built.shards));
        if (built.shards == NULL) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate owned weight shards");
        } else {
            built.shard_count = shard_count;
        }
    }
    if (status == M3_STATUS_OK && total_bindings != 0U) {
        built.bindings = calloc(total_bindings, sizeof(*built.bindings));
        if (built.bindings == NULL) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate weight bindings");
        } else {
            built.binding_count = total_bindings;
        }
    }
    for (shard_index = 0U;
         shard_index < shard_count && status == M3_STATUS_OK; ++shard_index) {
        const m3_weight_shard_source *source = order[shard_index].source;
        uint64_t payload_bytes = source->metadata->source_snapshot.file_size -
                                 source->metadata->data_section_offset;
        size_t tensor_index;

        status = m3_weight_copy_shard(&built.shards[shard_index], source,
                                      payload_bytes, error);
        for (tensor_index = 0U;
             tensor_index < source->metadata->tensor_count &&
             status == M3_STATUS_OK;
             ++tensor_index) {
            status = m3_weight_copy_binding(
                &built.bindings[output_index++],
                &source->metadata->tensors[tensor_index], shard_index, error);
        }
    }
    if (status == M3_STATUS_OK && total_bindings > 1U) {
        qsort(built.bindings, total_bindings, sizeof(*built.bindings),
              m3_weight_compare_bindings);
        for (output_index = 1U; output_index < total_bindings;
             ++output_index) {
            if (strcmp(built.bindings[output_index - 1U].name,
                       built.bindings[output_index].name) == 0) {
                status = m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "duplicate weight name '%s' across shards",
                    built.bindings[output_index].name);
                break;
            }
        }
    }
    free(order);
    if (status != M3_STATUS_OK) {
        m3_weight_table_dispose(&built);
        return status;
    }
    m3_weight_table_dispose(table);
    *table = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

const m3_weight_binding *m3_weight_table_find(const m3_weight_table *table,
                                              const char *name)
{
    size_t low = 0U;
    size_t high;

    if (table == NULL || name == NULL ||
        (table->binding_count != 0U && table->bindings == NULL)) {
        return NULL;
    }
    high = table->binding_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2U;
        int order = strcmp(name, table->bindings[middle].name);

        if (order == 0) {
            return &table->bindings[middle];
        }
        if (order < 0) {
            high = middle;
        } else {
            low = middle + 1U;
        }
    }
    return NULL;
}

static m3_status m3_weight_validate_requirements(
    const m3_weight_requirement *requirements, size_t requirement_count,
    m3_error *error)
{
    size_t left;

    if (requirement_count != 0U && requirements == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight requirements are null");
    }
    for (left = 0U; left < requirement_count; ++left) {
        size_t right;
        m3_status status;

        if (requirements[left].name == NULL ||
            requirements[left].name[0] == '\0') {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "required weight name is empty");
        }
        status = m3_weight_validate_metadata(&requirements[left].tensor,
                                             error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        for (right = left + 1U; right < requirement_count; ++right) {
            if (requirements[right].name != NULL &&
                strcmp(requirements[left].name,
                       requirements[right].name) == 0) {
                return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                    "duplicate required weight '%s'",
                                    requirements[left].name);
            }
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_weight_table_validate_required(
    const m3_weight_table *table, const m3_weight_requirement *requirements,
    size_t requirement_count, m3_error *error)
{
    size_t index;
    m3_status status;

    if (table == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight table is null");
    }
    status = m3_weight_validate_requirements(requirements,
                                             requirement_count, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    for (index = 0U; index < requirement_count; ++index) {
        const m3_weight_binding *binding =
            m3_weight_table_find(table, requirements[index].name);

        if (binding == NULL) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "missing required weight '%s'",
                                requirements[index].name);
        }
        if (!m3_weight_metadata_equal(&binding->tensor,
                                      &requirements[index].tensor)) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "weight '%s' has the wrong dtype or shape",
                                requirements[index].name);
        }
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_weight_table_validate_no_extra(
    const m3_weight_table *table, const m3_weight_requirement *requirements,
    size_t requirement_count, m3_error *error)
{
    size_t binding_index;
    m3_status status;

    if (table == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight table is null");
    }
    status = m3_weight_validate_requirements(requirements,
                                             requirement_count, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    for (binding_index = 0U; binding_index < table->binding_count;
         ++binding_index) {
        size_t requirement_index;
        bool found = false;

        for (requirement_index = 0U;
             requirement_index < requirement_count; ++requirement_index) {
            if (strcmp(table->bindings[binding_index].name,
                       requirements[requirement_index].name) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "unexpected extra weight '%s'",
                                table->bindings[binding_index].name);
        }
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}
