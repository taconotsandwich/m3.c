/* SPDX-License-Identifier: GPL-2.0-only */

#include "rvq_condition_test.h"

#include "m3_op_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

bool m3_rc_tensor(m3_op_test_fixture *fixture, m3_tensor_view *view,
                  m3_dtype dtype, uint8_t rank, const uint64_t *shape,
                  float initial)
{
    m3_tensor_metadata metadata;
    m3_error error;
    void *zeros;
    size_t index;
    bool ready;

    if (m3_tensor_metadata_init(&metadata, dtype, rank, shape, &error) !=
        M3_STATUS_OK) {
        return false;
    }
    zeros = calloc(1U, metadata.byte_count);
    if (zeros == NULL) {
        return false;
    }
    ready = m3_op_test_tensor(
        fixture, view, dtype, rank, shape, zeros);
    free(zeros);
    if (!ready) {
        return false;
    }
    for (index = 0U; index < metadata.element_count; ++index) {
        m3_rc_set(view, index, initial);
    }
    return true;
}

void m3_rc_set(m3_tensor_view *view, size_t flat_index, float value)
{
    m3_op_store_float(
        view, m3_op_element_offset(view, flat_index), value);
}

float m3_rc_get(const m3_tensor_view *view, size_t flat_index)
{
    return m3_op_load_float(
        view, m3_op_element_offset(view, flat_index));
}

bool m3_rc_progress(void *context, uint64_t completed, uint64_t total)
{
    m3_runtime_progress_log *log = context;

    if (log->call_count < 16U) {
        log->completed[log->call_count] = completed;
        log->total[log->call_count] = total;
    }
    ++log->call_count;
    return log->cancel_at == UINT64_MAX || completed < log->cancel_at;
}

static bool m3_rvq_test_weight(
    m3_rvq_test_fixture *fixture, const uint64_t *shape, uint8_t rank,
    const m3_tensor_view **output, float initial)
{
    m3_tensor_view *view = calloc(1U, sizeof(*view));

    if (view == NULL ||
        !m3_rc_tensor(&fixture->fixture, view, fixture->config.dtype,
                      rank, shape, initial)) {
        free(view);
        return false;
    }
    *output = view;
    return true;
}

static bool m3_rvq_test_layer(m3_rvq_test_fixture *fixture,
                              m3_rvq_layer_weights *layer)
{
    const uint64_t hidden[] = {fixture->config.hidden_size};
    const uint64_t matrix[] = {
        fixture->config.hidden_size, fixture->config.hidden_size
    };
    const uint64_t expanded[] = {
        fixture->config.intermediate_size, fixture->config.hidden_size
    };
    const uint64_t contracted[] = {
        fixture->config.hidden_size, fixture->config.intermediate_size
    };
    const m3_tensor_view **targets[] = {
        &layer->input_norm,
        &layer->query,
        &layer->key,
        &layer->value,
        &layer->attention_out,
        &layer->post_attention_norm,
        &layer->gate,
        &layer->up,
        &layer->down
    };
    const uint64_t *shapes[] = {
        hidden, matrix, matrix, matrix, matrix, hidden, expanded, expanded,
        contracted
    };
    const uint8_t ranks[] = {1U, 2U, 2U, 2U, 2U, 1U, 2U, 2U, 2U};
    size_t index;

    for (index = 0U; index < 9U; ++index) {
        if (!m3_rvq_test_weight(
                fixture, shapes[index], ranks[index], targets[index],
                0.0F)) {
            return false;
        }
    }
    for (index = 0U; index < fixture->config.hidden_size; ++index) {
        m3_rc_set((m3_tensor_view *)layer->input_norm, index, 1.0F);
        m3_rc_set((m3_tensor_view *)layer->post_attention_norm, index,
                  1.0F);
    }
    return true;
}

static void m3_rvq_test_free_weights(m3_rvq_test_fixture *fixture)
{
    size_t layer;
    size_t head;

    free((void *)fixture->weights.audio_embeddings);
    free((void *)fixture->weights.projection);
    free((void *)fixture->weights.position_embeddings);
    free((void *)fixture->weights.norm);
    for (layer = 0U; layer < fixture->config.layer_count; ++layer) {
        m3_rvq_layer_weights *weights = &fixture->weights.layers[layer];

        free((void *)weights->input_norm);
        free((void *)weights->query);
        free((void *)weights->key);
        free((void *)weights->value);
        free((void *)weights->attention_out);
        free((void *)weights->post_attention_norm);
        free((void *)weights->gate);
        free((void *)weights->up);
        free((void *)weights->down);
    }
    for (head = 0U; head < M3_RVQ_RESIDUAL_COUNT; ++head) {
        free((void *)fixture->weights.heads[head]);
    }
    (void)memset(&fixture->weights, 0, sizeof(fixture->weights));
}

bool m3_rvq_test_fixture_init(m3_rvq_test_fixture *fixture,
                              m3_dtype dtype)
{
    m3_backend *backend = NULL;
    m3_error error;

    if (m3_backend_create_host(&backend, &error) != M3_STATUS_OK) {
        return false;
    }
    return m3_rvq_test_fixture_init_backend(
        fixture, dtype, 2U, backend, true);
}

bool m3_rvq_test_fixture_init_backend(m3_rvq_test_fixture *fixture,
                                      m3_dtype dtype,
                                      uint32_t hidden_size,
                                      m3_backend *backend,
                                      bool owns_backend)
{
    const uint64_t audio[] = {
        (M3_RVQ_CODEBOOK_COUNT - 1U) * M3_RVQ_CODEBOOK_SIZE,
        hidden_size
    };
    const uint64_t matrix[] = {hidden_size, hidden_size};
    const uint64_t positions[] = {16U, hidden_size};
    const uint64_t hidden[] = {hidden_size};
    const uint64_t head[] = {M3_RVQ_CODEBOOK_SIZE, hidden_size};
    const uint64_t last[] = {2U, hidden_size};
    size_t index;

    if (fixture == NULL || backend == NULL || hidden_size < 2U ||
        hidden_size % 2U != 0U) {
        if (owns_backend) {
            m3_backend_free(backend);
        }
        return false;
    }
    (void)memset(fixture, 0, sizeof(*fixture));
    fixture->config = (m3_rvq_config){
        hidden_size, 1U, hidden_size / 2U, hidden_size, 16U, dtype
    };
    if (!m3_op_test_fixture_init_backend(
            &fixture->fixture, backend, owns_backend) ||
        !m3_rvq_test_weight(
            fixture, audio, 2U,
            &fixture->weights.audio_embeddings, 0.0F) ||
        !m3_rvq_test_weight(
            fixture, matrix, 2U,
            &fixture->weights.projection, 0.0F) ||
        !m3_rvq_test_weight(
            fixture, positions, 2U,
            &fixture->weights.position_embeddings,
            0.0F) ||
        !m3_rvq_test_weight(
            fixture, hidden, 1U,
            &fixture->weights.norm, 1.0F) ||
        !m3_rvq_test_layer(fixture, &fixture->weights.layers[0]) ||
        !m3_rc_tensor(&fixture->fixture, &fixture->last_hidden, dtype, 2U,
                      last, 0.0F) ||
        !m3_rc_tensor(&fixture->fixture, &fixture->semantic_embedding,
                      dtype, 1U, hidden, 0.0F)) {
        m3_rvq_test_fixture_dispose(fixture);
        return false;
    }
    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT; ++index) {
        if (!m3_rvq_test_weight(
                fixture, head, 2U,
                &fixture->weights.heads[index],
                0.0F)) {
            m3_rvq_test_fixture_dispose(fixture);
            return false;
        }
    }
    for (index = 0U; index < hidden_size; ++index) {
        m3_rc_set((m3_tensor_view *)fixture->weights.projection,
                  index * hidden_size + index, 1.0F);
    }
    for (index = 0U; index < 16U; ++index) {
        m3_rc_set((m3_tensor_view *)fixture->weights.position_embeddings,
                  index * hidden_size, (float)index * 0.25F);
        m3_rc_set((m3_tensor_view *)fixture->weights.position_embeddings,
                  index * hidden_size + 1U,
                  (float)index * 0.125F);
    }
    for (index = 0U; index < 7U * M3_RVQ_CODEBOOK_SIZE; ++index) {
        float book = (float)(index / M3_RVQ_CODEBOOK_SIZE + 1U);

        m3_rc_set((m3_tensor_view *)fixture->weights.audio_embeddings,
                  index * hidden_size, book);
        m3_rc_set((m3_tensor_view *)fixture->weights.audio_embeddings,
                  index * hidden_size + 1U, -0.5F * book);
    }
    m3_rc_set(&fixture->last_hidden, 0U, 3.0F);
    m3_rc_set(&fixture->last_hidden, 1U, 4.0F);
    m3_rc_set(&fixture->last_hidden, hidden_size, -2.0F);
    m3_rc_set(&fixture->last_hidden, hidden_size + 1U, 1.0F);
    m3_rc_set(&fixture->semantic_embedding, 0U, 0.5F);
    m3_rc_set(&fixture->semantic_embedding, 1U, 1.0F);
    return true;
}

void m3_rvq_test_fixture_dispose(m3_rvq_test_fixture *fixture)
{
    if (fixture == NULL) {
        return;
    }
    m3_rvq_test_free_weights(fixture);
    m3_op_test_fixture_dispose(&fixture->fixture);
    (void)memset(fixture, 0, sizeof(*fixture));
}

void m3_rvq_test_uniforms(float uniforms[M3_RVQ_RESIDUAL_COUNT],
                          uint32_t codes[M3_RVQ_RESIDUAL_COUNT])
{
    static const uint32_t selected[M3_RVQ_RESIDUAL_COUNT] = {
        0U, 17U, 128U, 255U, 512U, 777U, 1023U
    };
    size_t index;

    for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT; ++index) {
        codes[index] = selected[index];
        uniforms[index] = ((float)selected[index] + 0.25F) /
                          (float)M3_RVQ_CODEBOOK_SIZE;
    }
}

bool m3_condition_test_fixture_init(m3_condition_test_fixture *fixture)
{
    const uint64_t logits[] = {8U};
    const uint64_t scale[] = {1U};
    const uint64_t projection[] = {1U, 1U, 3U};
    const uint64_t bias[] = {1U};
    const uint64_t frames_shape[] = {1U, 3U, 8U, 1U};
    m3_storage *frame_storage = NULL;
    const size_t strides[] = {192U, 64U, 8U, 4U};
    m3_error error;
    size_t frame;
    size_t layer;

    static const float values[3][8] = {
        {1.0e8F, 1.0F, 1.0F, 1.0F, -1.0e8F, 1.0F, 1.0F, 1.0F},
        {1.0e8F, -1.0e8F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F},
        {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F}
    };

    (void)memset(fixture, 0, sizeof(*fixture));
    fixture->config = (m3_condition_config){1U, 8U, 1U, 3U, 2U};
    if (!m3_op_test_fixture_init(&fixture->fixture) ||
        !m3_rc_tensor(&fixture->fixture, &fixture->weight_views[0],
                      M3_DTYPE_F32, 1U, logits, 0.0F) ||
        !m3_rc_tensor(&fixture->fixture, &fixture->weight_views[1],
                      M3_DTYPE_F32, 1U, scale, 1.5F) ||
        !m3_rc_tensor(&fixture->fixture, &fixture->weight_views[2],
                      M3_DTYPE_F32, 3U, projection, 0.0F) ||
        !m3_rc_tensor(&fixture->fixture, &fixture->weight_views[3],
                      M3_DTYPE_F32, 1U, bias, 0.1F) ||
        !m3_op_test_storage(&fixture->fixture, 192U, &frame_storage) ||
        m3_tensor_view_strided(
            &fixture->frames, frame_storage, M3_DTYPE_BF16, 4U,
            frames_shape, strides, 0U, &error) != M3_STATUS_OK) {
        m3_condition_test_fixture_dispose(fixture);
        return false;
    }
    fixture->weights.layer_weight_logits = &fixture->weight_views[0];
    fixture->weights.layer_scale = &fixture->weight_views[1];
    fixture->weights.projection = &fixture->weight_views[2];
    fixture->weights.bias = &fixture->weight_views[3];
    m3_rc_set(&fixture->weight_views[2], 0U, 1.0F);
    m3_rc_set(&fixture->weight_views[2], 1U, -0.5F);
    m3_rc_set(&fixture->weight_views[2], 2U, 0.25F);
    for (frame = 0U; frame < 3U; ++frame) {
        for (layer = 0U; layer < 8U; ++layer) {
            m3_rc_set(&fixture->frames, frame * 8U + layer,
                      values[frame][layer]);
        }
    }
    return true;
}

void m3_condition_test_fixture_dispose(
    m3_condition_test_fixture *fixture)
{
    if (fixture == NULL) {
        return;
    }
    m3_op_test_fixture_dispose(&fixture->fixture);
    (void)memset(fixture, 0, sizeof(*fixture));
}

float m3_condition_test_mixed(const m3_tensor_view *frames,
                              size_t frame_index)
{
    float mixed = 0.0F;
    size_t layer;

    for (layer = 0U; layer < 8U; ++layer) {
        float product = m3_rc_get(frames, frame_index * 8U + layer) *
                        0.125F;

        mixed = layer == 0U ? product : mixed + product;
    }
    return mixed;
}
