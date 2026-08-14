/* SPDX-License-Identifier: GPL-2.0-only */

#include "qwen_runtime_fixture.h"

#include "m3_op_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool m3_qwen_test_bf16(
    m3_qwen_test_fixture *fixture, m3_tensor_view *view, uint8_t rank,
    const uint64_t *shape, const float *values)
{
    m3_tensor_metadata metadata;
    uint16_t *zeros;
    m3_error error;
    size_t index;

    if (m3_tensor_metadata_init(&metadata, M3_DTYPE_BF16, rank, shape,
                                &error) != M3_STATUS_OK) {
        return false;
    }
    zeros = calloc(metadata.element_count, sizeof(*zeros));
    if (zeros == NULL ||
        !m3_op_test_tensor(&fixture->tensors, view, M3_DTYPE_BF16, rank,
                           shape, zeros)) {
        free(zeros);
        return false;
    }
    free(zeros);
    for (index = 0U; index < metadata.element_count; ++index) {
        m3_op_store_float(view, m3_op_element_offset(view, index),
                          values[index]);
    }
    return true;
}

static bool m3_qwen_test_global_weights(m3_qwen_test_fixture *fixture)
{
    static const float embedding[] = {
        1.0F, 0.5F, -0.5F, 2.0F,
        -1.0F, 1.0F, 0.25F, 0.5F,
        0.75F, -0.25F, 1.5F, -1.0F,
        0.5F, 1.0F, -1.5F, 0.25F,
        1.25F, 0.5F, 0.75F, -0.5F,
        -0.5F, 1.5F, 0.5F, 1.0F,
        0.25F, -1.0F, 1.25F, 0.75F,
        2.0F, -0.5F, 0.5F, -1.0F
    };
    static const float final_norm[] = {1.0F, 1.25F, 0.75F, 0.5F};
    static const float head[] = {
        0.5F, 0.0F, 0.0F, 0.0F,
        1.0F, -0.5F, 0.25F, 0.75F,
        0.0F, 0.5F, 0.0F, 0.0F,
        0.0F, 0.0F, 0.5F, 0.0F,
        -0.75F, 0.5F, 1.0F, 0.25F,
        0.25F, 1.25F, -0.5F, 0.75F,
        1.5F, -0.25F, 0.5F, -1.0F,
        0.0F, 0.0F, 0.0F, 0.5F
    };
    const uint64_t embedding_shape[] = {8U, 4U};
    const uint64_t norm_shape[] = {4U};
    const uint64_t head_shape[] = {8U, 4U};
    m3_tensor_view *embedding_view = &fixture->weight_views[0];
    m3_tensor_view *norm_view = &fixture->weight_views[1];
    m3_tensor_view *head_view = &fixture->weight_views[2];

    if (!m3_qwen_test_bf16(fixture, embedding_view, 2U, embedding_shape,
                           embedding) ||
        !m3_qwen_test_bf16(fixture, norm_view, 1U, norm_shape,
                           final_norm) ||
        !m3_qwen_test_bf16(fixture, head_view, 2U, head_shape, head)) {
        return false;
    }
    fixture->weights.embedding = embedding_view;
    fixture->weights.final_norm = norm_view;
    fixture->weights.head = head_view;
    return true;
}

static bool m3_qwen_test_layer_weights(m3_qwen_test_fixture *fixture)
{
    static const float input_norm[] = {1.0F, 0.75F, 1.25F, 0.5F};
    static const float query[] = {
        1.0F, 0.25F, 0.0F, -0.25F,
        0.0F, 0.75F, 0.5F, 0.0F,
        -0.5F, 0.0F, 1.0F, 0.25F,
        0.25F, -0.5F, 0.0F, 1.0F
    };
    static const float key[] = {
        0.75F, -0.25F, 0.5F, 0.0F,
        0.0F, 0.5F, -0.5F, 1.0F
    };
    static const float value[] = {
        0.5F, 0.25F, -0.75F, 1.0F,
        -0.25F, 1.0F, 0.5F, 0.25F
    };
    static const float output[] = {
        0.5F, 0.0F, 0.25F, -0.25F,
        -0.5F, 0.75F, 0.0F, 0.25F,
        0.25F, -0.25F, 1.0F, 0.0F,
        0.0F, 0.5F, -0.5F, 0.75F
    };
    static const float query_norm[] = {1.0F, 0.75F};
    static const float key_norm[] = {0.5F, 1.25F};
    static const float post_norm[] = {0.75F, 1.0F, 0.5F, 1.25F};
    static const float gate[] = {
        0.5F, -0.25F, 0.75F, 0.0F,
        0.0F, 1.0F, -0.5F, 0.25F,
        0.75F, 0.0F, 0.25F, -0.5F,
        -0.25F, 0.5F, 0.0F, 1.0F
    };
    static const float up[] = {
        1.0F, 0.5F, 0.0F, -0.25F,
        -0.5F, 0.25F, 1.0F, 0.0F,
        0.0F, -0.75F, 0.5F, 1.0F,
        0.25F, 0.0F, -0.5F, 0.75F
    };
    static const float down[] = {
        0.5F, -0.25F, 0.0F, 0.75F,
        0.25F, 0.5F, -0.5F, 0.0F,
        -0.75F, 0.0F, 0.5F, 0.25F,
        0.0F, 1.0F, 0.25F, -0.5F
    };
    const float *values[] = {
        input_norm, query, key, value, output, query_norm, key_norm,
        post_norm, gate, up, down
    };
    const uint8_t ranks[] = {1U, 2U, 2U, 2U, 2U, 1U,
                             1U, 1U, 2U, 2U, 2U};
    const uint64_t shapes[][2] = {
        {4U, 0U}, {4U, 4U}, {2U, 4U}, {2U, 4U}, {4U, 4U},
        {2U, 0U}, {2U, 0U}, {4U, 0U}, {4U, 4U}, {4U, 4U},
        {4U, 4U}
    };
    const m3_tensor_view **slots[] = {
        &fixture->weights.layers[0].input_norm,
        &fixture->weights.layers[0].query_weight,
        &fixture->weights.layers[0].key_weight,
        &fixture->weights.layers[0].value_weight,
        &fixture->weights.layers[0].output_weight,
        &fixture->weights.layers[0].query_norm,
        &fixture->weights.layers[0].key_norm,
        &fixture->weights.layers[0].post_attention_norm,
        &fixture->weights.layers[0].gate_weight,
        &fixture->weights.layers[0].up_weight,
        &fixture->weights.layers[0].down_weight
    };
    size_t index;

    for (index = 0U; index < 11U; ++index) {
        m3_tensor_view *view = &fixture->weight_views[3U + index];

        if (!m3_qwen_test_bf16(fixture, view, ranks[index], shapes[index],
                               values[index])) {
            return false;
        }
        *slots[index] = view;
    }
    return true;
}

bool m3_qwen_test_fixture_init(m3_qwen_test_fixture *fixture,
                               uint64_t capacity)
{
    m3_backend *backend = NULL;
    m3_error error;

    if (m3_backend_create_host(&backend, &error) != M3_STATUS_OK) {
        return false;
    }
    return m3_qwen_test_fixture_init_backend(
        fixture, capacity, backend, true);
}

bool m3_qwen_test_fixture_init_backend(m3_qwen_test_fixture *fixture,
                                       uint64_t capacity,
                                       m3_backend *backend,
                                       bool owns_backend)
{
    m3_error error;
    size_t index;

    if (fixture == NULL || backend == NULL) {
        if (owns_backend) {
            m3_backend_free(backend);
        }
        return false;
    }
    (void)memset(fixture, 0, sizeof(*fixture));
    m3_qwen_cache_init(&fixture->cache);
    m3_runtime_workspace_init(&fixture->rope);
    m3_qwen_forward_state_init(&fixture->forward);
    if (!m3_op_test_fixture_init_backend(
            &fixture->tensors, backend, owns_backend)) {
        return false;
    }
    fixture->dimensions.vocab_size = 8U;
    fixture->dimensions.hidden_size = 4U;
    fixture->dimensions.layer_count = 1U;
    fixture->dimensions.query_heads = 2U;
    fixture->dimensions.key_value_heads = 1U;
    fixture->dimensions.head_dimension = 2U;
    fixture->dimensions.intermediate_size = 4U;
    fixture->dimensions.eos_token_id = 1U;
    fixture->dimensions.semantic_token_start = 4U;
    fixture->dimensions.semantic_token_count = 3U;
    fixture->dimensions.rms_epsilon = 1.0e-6F;
    fixture->dimensions.rope_theta = 100.0F;
    if (!m3_qwen_test_global_weights(fixture) ||
        !m3_qwen_test_layer_weights(fixture) ||
        m3_qwen_cache_build(&fixture->cache, fixture->tensors.backend,
                            &fixture->dimensions, capacity, &error) !=
            M3_STATUS_OK ||
        m3_qwen_rope_build(&fixture->rope, fixture->tensors.backend,
                           &fixture->dimensions, capacity, &error) !=
            M3_STATUS_OK) {
        m3_qwen_test_fixture_dispose(fixture);
        return false;
    }
    for (index = 0U; index < fixture->cache.workspace.count; ++index) {
        (void)memset(m3_storage_data(fixture->cache.workspace.storages[index]),
                     0, m3_storage_size(
                            fixture->cache.workspace.storages[index]));
    }
    m3_command_executor_init(&fixture->executor, fixture->tensors.backend);
    fixture->forward.dimensions = &fixture->dimensions;
    fixture->forward.weights = &fixture->weights;
    fixture->forward.backend = fixture->tensors.backend;
    fixture->forward.executor = &fixture->executor;
    fixture->forward.cache_views = fixture->cache.workspace.views;
    fixture->forward.cosines = &fixture->rope.views[0];
    fixture->forward.sines = &fixture->rope.views[1];
    fixture->forward.cache_capacity = capacity;
    return true;
}

void m3_qwen_test_fixture_dispose(m3_qwen_test_fixture *fixture)
{
    if (fixture == NULL) {
        return;
    }
    m3_qwen_forward_state_dispose(&fixture->forward);
    m3_command_executor_dispose(&fixture->executor);
    m3_runtime_workspace_dispose(&fixture->rope);
    m3_qwen_cache_dispose(&fixture->cache);
    m3_op_test_fixture_dispose(&fixture->tensors);
    (void)memset(fixture, 0, sizeof(*fixture));
}

bool m3_qwen_test_ids(m3_qwen_test_fixture *fixture,
                      m3_tensor_view *view, uint64_t sequence,
                      const int32_t *values)
{
    const uint64_t shape[] = {2U, sequence};

    return m3_op_test_tensor(&fixture->tensors, view, M3_DTYPE_I32, 2U,
                             shape, values);
}

bool m3_qwen_test_feedback(m3_qwen_test_fixture *fixture,
                           m3_tensor_view *view, const float *values)
{
    const uint64_t shape[] = {2U, 1U, fixture->dimensions.hidden_size};

    return m3_qwen_test_bf16(fixture, view, 3U, shape, values);
}

uint16_t m3_qwen_test_bf16_at(const m3_tensor_view *view, size_t index)
{
    const uint8_t *data = m3_storage_const_data(view->storage);
    const uint16_t *values = (const uint16_t *)(data + view->byte_offset);

    return values[index];
}

float m3_qwen_test_f32_at(const m3_tensor_view *view, size_t index)
{
    const uint8_t *data = m3_storage_const_data(view->storage);
    const float *values = (const float *)(data + view->byte_offset);

    return values[index];
}
