/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_weights.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
                            "weight name and shard path cannot be empty");
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

static int m3_weight_compare(const void *left, const void *right)
{
    const m3_weight_binding *left_binding = left;
    const m3_weight_binding *right_binding = right;

    return strcmp(left_binding->name, right_binding->name);
}

void m3_weight_table_init(m3_weight_table *table)
{
    if (table != NULL) {
        table->bindings = NULL;
        table->binding_count = 0U;
    }
}

void m3_weight_table_dispose(m3_weight_table *table)
{
    size_t index;

    if (table == NULL) {
        return;
    }
    for (index = 0U; index < table->binding_count; ++index) {
        free(table->bindings[index].name);
        free(table->bindings[index].shard_path);
    }
    free(table->bindings);
    m3_weight_table_init(table);
}

static m3_status m3_weight_binding_copy(
    m3_weight_binding *binding, const char *shard_path,
    const m3_safetensors_metadata *metadata,
    const m3_safetensors_tensor *tensor, m3_error *error)
{
    uint64_t relative_bytes;
    m3_status status;

    if (binding == NULL || metadata == NULL || tensor == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight binding source is null");
    }
    (void)memset(binding, 0, sizeof(*binding));
    if (tensor->name == NULL || tensor->name[0] == '\0') {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "inspected weight name is empty");
    }
    status = m3_weight_validate_metadata(&tensor->tensor, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (tensor->data_end < tensor->data_start) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "weight '%s' has a reversed byte range",
                            tensor->name);
    }
    relative_bytes = tensor->data_end - tensor->data_start;
    if (relative_bytes != (uint64_t)tensor->tensor.byte_count) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "weight '%s' range disagrees with metadata",
                            tensor->name);
    }
    if (metadata->data_section_offset > UINT64_MAX - tensor->data_start ||
        metadata->data_section_offset > UINT64_MAX - tensor->data_end) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "weight '%s' absolute file range overflows",
                            tensor->name);
    }
    status = m3_weight_copy_string(tensor->name, &binding->name, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    status = m3_weight_copy_string(shard_path, &binding->shard_path, error);
    if (status != M3_STATUS_OK) {
        free(binding->name);
        binding->name = NULL;
        return status;
    }
    binding->tensor = tensor->tensor;
    binding->absolute_file_start =
        metadata->data_section_offset + tensor->data_start;
    binding->absolute_file_end =
        metadata->data_section_offset + tensor->data_end;
    return M3_STATUS_OK;
}

m3_status m3_weight_table_build(m3_weight_table *table,
                                const m3_weight_shard *shards,
                                size_t shard_count, m3_error *error)
{
    m3_weight_table built;
    size_t total = 0U;
    size_t shard_index;
    size_t output_index = 0U;
    m3_status status = M3_STATUS_OK;

    if (table == NULL || (shard_count != 0U && shards == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight table and shards are required");
    }
    m3_weight_table_init(&built);
    for (shard_index = 0U; shard_index < shard_count; ++shard_index) {
        const m3_safetensors_metadata *metadata = shards[shard_index].metadata;

        if (shards[shard_index].shard_path == NULL ||
            shards[shard_index].shard_path[0] == '\0' || metadata == NULL ||
            (metadata->tensor_count != 0U && metadata->tensors == NULL)) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "weight shard descriptor is incomplete");
        }
        if (metadata->tensor_count > SIZE_MAX - total) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "weight binding count overflows");
        }
        total += metadata->tensor_count;
    }
    if (total > SIZE_MAX / sizeof(*built.bindings)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "weight binding allocation overflows");
    }
    if (total != 0U) {
        built.bindings = calloc(total, sizeof(*built.bindings));
        if (built.bindings == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate weight binding table");
        }
    }
    built.binding_count = total;
    for (shard_index = 0U; shard_index < shard_count; ++shard_index) {
        const m3_safetensors_metadata *metadata = shards[shard_index].metadata;
        size_t tensor_index;

        for (tensor_index = 0U; tensor_index < metadata->tensor_count;
             ++tensor_index) {
            status = m3_weight_binding_copy(
                &built.bindings[output_index],
                shards[shard_index].shard_path, metadata,
                &metadata->tensors[tensor_index], error);
            if (status != M3_STATUS_OK) {
                m3_weight_table_dispose(&built);
                return status;
            }
            ++output_index;
        }
    }
    if (total > 1U) {
        qsort(built.bindings, total, sizeof(*built.bindings),
              m3_weight_compare);
        for (output_index = 1U; output_index < total; ++output_index) {
            if (strcmp(built.bindings[output_index - 1U].name,
                       built.bindings[output_index].name) == 0) {
                status = m3_error_set(
                    error, M3_STATUS_INVALID_FORMAT,
                    "duplicate weight name '%s' across shards",
                    built.bindings[output_index].name);
                m3_weight_table_dispose(&built);
                return status;
            }
        }
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
