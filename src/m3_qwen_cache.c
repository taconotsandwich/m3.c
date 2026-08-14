/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_qwen_internal.h"

#include "m3_op_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static m3_status m3_qwen_dimensions_validate(
    const m3_qwen_dimensions *dimensions, m3_error *error)
{
    if (dimensions == NULL || dimensions->hidden_size == 0U ||
        dimensions->layer_count == 0U ||
        dimensions->layer_count > M3_QWEN_LAYER_COUNT ||
        dimensions->query_heads == 0U ||
        dimensions->key_value_heads == 0U ||
        dimensions->head_dimension == 0U ||
        dimensions->head_dimension % 2U != 0U ||
        dimensions->query_heads % dimensions->key_value_heads != 0U ||
        dimensions->query_heads > UINT64_MAX /
                                      dimensions->head_dimension ||
        dimensions->query_heads * dimensions->head_dimension !=
            dimensions->hidden_size) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen dimensions are invalid");
    }
    return M3_STATUS_OK;
}

m3_status m3_qwen_cache_spec(const m3_qwen_dimensions *dimensions,
                             uint64_t capacity, size_t index,
                             m3_runtime_tensor_spec *spec,
                             m3_error *error)
{
    size_t count;
    m3_status status = m3_qwen_dimensions_validate(dimensions, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (capacity == 0U || dimensions->layer_count > SIZE_MAX / 2U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen cache capacity is zero or count is invalid");
    }
    count = (size_t)dimensions->layer_count * 2U;
    if (spec == NULL || index >= count) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "Qwen cache storage index is out of range");
    }
    (void)memset(spec, 0, sizeof(*spec));
    spec->dtype = M3_DTYPE_BF16;
    spec->rank = 4U;
    spec->shape[0] = capacity;
    spec->shape[1] = 2U;
    spec->shape[2] = dimensions->key_value_heads;
    spec->shape[3] = dimensions->head_dimension;
    spec->alignment = 64U;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_qwen_cache_measure(const m3_qwen_dimensions *dimensions,
                                uint64_t capacity, size_t *storage_count,
                                size_t *byte_count, m3_error *error)
{
    m3_runtime_tensor_spec spec;
    m3_tensor_metadata metadata;
    size_t count;
    size_t total;
    m3_status status;

    (void)memset(&spec, 0, sizeof(spec));
    (void)memset(&metadata, 0, sizeof(metadata));
    if (storage_count == NULL || byte_count == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen cache measurement outputs are required");
    }
    status = m3_qwen_cache_spec(dimensions, capacity, 0U, &spec, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    count = (size_t)dimensions->layer_count * 2U;
    if (count == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen cache storage count is zero");
    }
    status = m3_tensor_metadata_init(&metadata, spec.dtype, spec.rank,
                                     spec.shape, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (metadata.byte_count > SIZE_MAX / count) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "Qwen cache byte total overflows");
    }
    total = metadata.byte_count * count;
    *storage_count = count;
    *byte_count = total;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

void m3_qwen_cache_init(m3_qwen_cache *cache)
{
    if (cache != NULL) {
        (void)memset(cache, 0, sizeof(*cache));
        m3_runtime_workspace_init(&cache->workspace);
    }
}

void m3_qwen_cache_dispose(m3_qwen_cache *cache)
{
    if (cache == NULL) {
        return;
    }
    m3_runtime_workspace_dispose(&cache->workspace);
    m3_qwen_cache_init(cache);
}

m3_status m3_qwen_cache_build(m3_qwen_cache *cache, m3_backend *backend,
                              const m3_qwen_dimensions *dimensions,
                              uint64_t capacity, m3_error *error)
{
    m3_qwen_cache built;
    m3_runtime_tensor_spec *specs = NULL;
    size_t count = 0U;
    size_t bytes = 0U;
    size_t index;
    m3_status status;

    if (cache == NULL || backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen cache and backend are required");
    }
    status = m3_qwen_cache_measure(dimensions, capacity, &count, &bytes,
                                   error);
    if (status == M3_STATUS_OK &&
        count > SIZE_MAX / sizeof(*specs)) {
        status = m3_error_set(error, M3_STATUS_OVERFLOW,
                              "Qwen cache specs allocation overflows");
    }
    if (status == M3_STATUS_OK) {
        specs = calloc(count, sizeof(*specs));
        if (specs == NULL) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate Qwen cache specs");
        }
    }
    for (index = 0U; index < count && status == M3_STATUS_OK; ++index) {
        status = m3_qwen_cache_spec(dimensions, capacity, index,
                                    &specs[index], error);
    }
    m3_qwen_cache_init(&built);
    if (status == M3_STATUS_OK) {
        status = m3_runtime_workspace_build(
            &built.workspace, backend, specs, count, error);
    }
    free(specs);
    if (status != M3_STATUS_OK) {
        m3_qwen_cache_dispose(&built);
        return status;
    }
    built.capacity = capacity;
    built.byte_count = bytes;
    m3_qwen_cache_dispose(cache);
    *cache = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

static void m3_qwen_rope_fill(const m3_qwen_dimensions *dimensions,
                              uint64_t capacity, m3_tensor_view *cosines,
                              m3_tensor_view *sines)
{
    uint64_t position;
    uint64_t pairs = dimensions->head_dimension / 2U;

    for (position = 0U; position < capacity; ++position) {
        uint64_t pair;

        for (pair = 0U; pair < pairs; ++pair) {
            float exponent = (float)(pair * 2U) /
                             (float)dimensions->head_dimension;
            float inverse_frequency =
                1.0F / powf(dimensions->rope_theta, exponent);
            float angle = (float)position * inverse_frequency;
            size_t flat = (size_t)(position * pairs + pair);

            m3_op_store_float(cosines, m3_op_element_offset(cosines, flat),
                              cosf(angle));
            m3_op_store_float(sines, m3_op_element_offset(sines, flat),
                              sinf(angle));
        }
    }
}

m3_status m3_qwen_rope_build(m3_runtime_workspace *rope,
                             m3_backend *backend,
                             const m3_qwen_dimensions *dimensions,
                             uint64_t capacity, m3_error *error)
{
    m3_runtime_workspace built;
    m3_runtime_tensor_spec specs[2];
    uint64_t shape[2];
    m3_status status = m3_qwen_dimensions_validate(dimensions, error);

    if (rope == NULL || backend == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen RoPE workspace and backend are required");
    }
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (capacity == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "Qwen RoPE capacity is zero");
    }
    shape[0] = capacity;
    shape[1] = dimensions->head_dimension / 2U;
    (void)memset(specs, 0, sizeof(specs));
    specs[0].dtype = M3_DTYPE_BF16;
    specs[0].rank = 2U;
    specs[0].shape[0] = shape[0];
    specs[0].shape[1] = shape[1];
    specs[0].alignment = 64U;
    specs[1] = specs[0];
    m3_runtime_workspace_init(&built);
    status = m3_runtime_workspace_build(&built, backend, specs, 2U, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    m3_qwen_rope_fill(dimensions, capacity, &built.views[0],
                      &built.views[1]);
    m3_runtime_workspace_dispose(rope);
    *rope = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}
