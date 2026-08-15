/* SPDX-License-Identifier: GPL-2.0-only */

#include "flow_runtime_test.h"

#include "rvq_condition_test.h"

#include <string.h>

static bool m3_flow_test_weight(
    m3_flow_test_fixture *fixture, const m3_tensor_view **target,
    uint8_t rank, const uint64_t *shape, float initial)
{
    m3_tensor_view *view;

    if (fixture->weight_count >=
        sizeof(fixture->weight_views) / sizeof(fixture->weight_views[0])) {
        return false;
    }
    view = &fixture->weight_views[fixture->weight_count];
    if (!m3_rc_tensor(&fixture->fixture, view, M3_DTYPE_F32, rank, shape,
                      initial)) {
        return false;
    }
    ++fixture->weight_count;
    *target = view;
    return true;
}

static bool m3_flow_test_root_weights(m3_flow_test_fixture *fixture)
{
    const m3_flow_config *config = &fixture->config;
    const uint64_t inner = (uint64_t)config->attention_heads *
                           config->head_dimension;
    const uint64_t concat = (uint64_t)config->latent_channels * 2U +
                            config->condition_dimension;
    const uint64_t time_projection[] = {
        config->fourier_dimension / 2U, 1U
    };
    const uint64_t time_in[] = {inner, config->fourier_dimension};
    const uint64_t vector[] = {inner};
    const uint64_t time_out[] = {inner, inner};
    const uint64_t preprocess[] = {concat, concat, 1U};
    const uint64_t input_projection[] = {inner, concat};
    const uint64_t output_projection[] = {
        config->latent_channels, inner
    };
    const uint64_t postprocess[] = {
        config->latent_channels, config->latent_channels, 1U
    };

    return m3_flow_test_weight(
               fixture, &fixture->weights.time_projection, 2U,
               time_projection, 0.0F) &&
           m3_flow_test_weight(
               fixture, &fixture->weights.time_linear_in, 2U, time_in,
               0.0F) &&
           m3_flow_test_weight(
               fixture, &fixture->weights.time_linear_in_bias, 1U,
               vector, 0.0F) &&
           m3_flow_test_weight(
               fixture, &fixture->weights.time_linear_out, 2U, time_out,
               0.0F) &&
           m3_flow_test_weight(
               fixture, &fixture->weights.time_linear_out_bias, 1U,
               vector, 0.0F) &&
           m3_flow_test_weight(
               fixture, &fixture->weights.preprocess_convolution, 3U,
               preprocess, 0.0F) &&
           m3_flow_test_weight(
               fixture, &fixture->weights.input_projection, 2U,
               input_projection, 0.0F) &&
           m3_flow_test_weight(
               fixture, &fixture->weights.output_projection, 2U,
               output_projection, 0.0F) &&
           m3_flow_test_weight(
               fixture, &fixture->weights.postprocess_convolution, 3U,
               postprocess, 0.0F);
}

static bool m3_flow_test_layer_weights(m3_flow_test_fixture *fixture)
{
    const m3_flow_config *config = &fixture->config;
    m3_flow_layer_weights *layer = &fixture->weights.layers[0];
    const uint64_t inner = (uint64_t)config->attention_heads *
                           config->head_dimension;
    const uint64_t vector[] = {inner};
    const uint64_t square[] = {inner, inner};
    const uint64_t ff_in[] = {
        (uint64_t)config->feed_forward_dimension * 2U, inner
    };
    const uint64_t ff_in_bias[] = {
        (uint64_t)config->feed_forward_dimension * 2U
    };
    const uint64_t ff_out[] = {
        inner, config->feed_forward_dimension
    };
    bool ready;

#define M3_FLOW_TEST_WEIGHT(member, rank, shape, initial)                   \
    m3_flow_test_weight(fixture, &layer->member, rank, shape, initial)
    ready = M3_FLOW_TEST_WEIGHT(norm1_scale, 1U, vector, 1.0F) &&
            M3_FLOW_TEST_WEIGHT(norm1_bias, 1U, vector, 0.0F) &&
            M3_FLOW_TEST_WEIGHT(norm2_scale, 1U, vector, 1.0F) &&
            M3_FLOW_TEST_WEIGHT(norm2_bias, 1U, vector, 0.0F) &&
            M3_FLOW_TEST_WEIGHT(query, 2U, square, 0.0F) &&
            M3_FLOW_TEST_WEIGHT(key, 2U, square, 0.0F) &&
            M3_FLOW_TEST_WEIGHT(value, 2U, square, 0.0F) &&
            M3_FLOW_TEST_WEIGHT(attention_out, 2U, square, 0.0F) &&
            M3_FLOW_TEST_WEIGHT(feed_forward_in, 2U, ff_in, 0.0F) &&
            M3_FLOW_TEST_WEIGHT(feed_forward_in_bias, 1U, ff_in_bias,
                                0.0F) &&
            M3_FLOW_TEST_WEIGHT(feed_forward_out, 2U, ff_out, 0.0F) &&
            M3_FLOW_TEST_WEIGHT(feed_forward_out_bias, 1U, vector, 0.0F);
#undef M3_FLOW_TEST_WEIGHT
    return ready;
}

static bool m3_flow_test_condition_weights(m3_flow_test_fixture *fixture)
{
    const m3_flow_config *config = &fixture->config;
    const uint64_t layers[] = {config->condition.layer_count};
    const uint64_t scale[] = {1U};
    const uint64_t projection[] = {
        config->condition.output_size, config->condition.hidden_size, 3U
    };
    const uint64_t bias[] = {config->condition.output_size};
    size_t channel;
    bool ready =
        m3_flow_test_weight(
            fixture, &fixture->condition_weights.layer_weight_logits,
            1U, layers, 0.0F) &&
        m3_flow_test_weight(
            fixture, &fixture->condition_weights.layer_scale, 1U, scale,
            1.0F) &&
        m3_flow_test_weight(
            fixture, &fixture->condition_weights.projection, 3U,
            projection, 0.0F) &&
        m3_flow_test_weight(
            fixture, &fixture->condition_weights.bias, 1U, bias, 0.0F);

    for (channel = 0U; channel < config->condition.output_size && ready;
         ++channel) {
        size_t offset =
            (channel * config->condition.hidden_size + channel) * 3U + 1U;

        m3_rc_set(
            (m3_tensor_view *)fixture->condition_weights.projection,
            offset, 1.0F);
    }
    return ready;
}

static bool m3_flow_test_frames(m3_flow_test_fixture *fixture,
                                uint64_t frame_count)
{
    const uint64_t shape[] = {
        1U, frame_count, fixture->config.condition.layer_count,
        fixture->config.condition.hidden_size
    };
    size_t index;
    bool ready = m3_rc_tensor(
        &fixture->fixture, &fixture->frames, M3_DTYPE_BF16, 4U, shape,
        0.0F);

    for (index = 0U; index < fixture->frames.metadata.element_count &&
                     ready; ++index) {
        m3_rc_set(&fixture->frames, index,
                  (float)((int)(index % 11U) - 5) * 0.125F);
    }
    return ready;
}

static bool m3_flow_test_fixture_build(
    m3_flow_test_fixture *fixture, uint64_t frame_count,
    m3_backend *backend, bool music3)
{
    bool ready;

    if (fixture == NULL || frame_count == 0U) {
        return false;
    }
    (void)memset(fixture, 0, sizeof(*fixture));
    ready = backend == NULL
                ? m3_op_test_fixture_init(&fixture->fixture)
                : m3_op_test_fixture_init_backend(
                      &fixture->fixture, backend, false);
    if (music3) {
        fixture->config = (m3_flow_config){
            M3_FLOW_LATENT_CHANNELS, 2U, 1U, 1U, 4U, 3U, 2U, 4U,
            M3_FLOW_CHUNK_FRAMES, M3_FLOW_CHUNK_HOP,
            M3_FLOW_CARRY_LENGTH, M3_FLOW_INFERENCE_STEPS,
            M3_FLOW_MAX_FRAMES, 1.7F, 1.0e-5F, 10000.0F,
            {2U, 2U, 2U, 441U, 128U}
        };
    } else {
        fixture->config = (m3_flow_config){
            2U, 2U, 1U, 1U, 4U, 3U, 2U, 4U,
            4U, 2U, 1U, 2U, 10U, 1.7F, 1.0e-5F, 10000.0F,
            {2U, 2U, 2U, 1U, 1U}
        };
    }
    if (ready) {
        ready = m3_flow_test_root_weights(fixture) &&
                m3_flow_test_layer_weights(fixture) &&
                m3_flow_test_condition_weights(fixture) &&
                m3_flow_test_frames(fixture, frame_count);
    }
    if (!ready) {
        m3_flow_test_fixture_dispose(fixture);
    }
    return ready;
}

bool m3_flow_test_fixture_init(m3_flow_test_fixture *fixture,
                               uint64_t frame_count)
{
    return m3_flow_test_fixture_build(
        fixture, frame_count, NULL, false);
}

bool m3_flow_test_fixture_init_music3(
    m3_flow_test_fixture *fixture, uint64_t frame_count,
    m3_backend *backend)
{
    return m3_flow_test_fixture_build(
        fixture, frame_count, backend, true);
}

void m3_flow_test_fixture_dispose(m3_flow_test_fixture *fixture)
{
    if (fixture == NULL) {
        return;
    }
    m3_op_test_fixture_dispose(&fixture->fixture);
    (void)memset(fixture, 0, sizeof(*fixture));
}

bool m3_flow_test_strided_frames(m3_flow_test_fixture *fixture,
                                 uint64_t frame_count,
                                 m3_tensor_view *frames)
{
    const uint64_t shape[] = {
        1U, frame_count, fixture->config.condition.layer_count,
        fixture->config.condition.hidden_size
    };
    const size_t strides[] = {
        (size_t)frame_count * 32U, 32U, 12U, 2U
    };
    m3_storage *storage = NULL;
    m3_error error;
    size_t index;
    bool ready = m3_op_test_storage(
        &fixture->fixture, (size_t)frame_count * 32U, &storage);

    m3_tensor_view_init(frames);
    if (ready) {
        ready = m3_tensor_view_strided(
                    frames, storage, M3_DTYPE_BF16, 4U, shape, strides,
                    0U, &error) == M3_STATUS_OK;
    }
    for (index = 0U; ready && index < frames->metadata.element_count;
         ++index) {
        m3_rc_set(frames, index,
                  (float)((int)(index % 11U) - 5) * 0.125F);
    }
    return ready;
}

bool m3_flow_test_run_init(m3_flow_test_fixture *fixture,
                           uint64_t maximum_length, m3_flow_run *run)
{
    m3_runtime_tensor_spec specs[M3_FLOW_WORKSPACE_COUNT];
    m3_error error;

    (void)memset(run, 0, sizeof(*run));
    run->backend = fixture->fixture.backend;
    run->config = &fixture->config;
    run->weights = &fixture->weights;
    run->maximum_length = maximum_length;
    m3_runtime_workspace_init(&run->workspace);
    m3_command_executor_init(&run->executor, run->backend);
    m3_flow_workspace_specs(&fixture->config, maximum_length, specs);
    return m3_runtime_workspace_build(
               &run->workspace, run->backend, specs,
               M3_FLOW_WORKSPACE_COUNT, &error) == M3_STATUS_OK;
}

void m3_flow_test_run_dispose(m3_flow_run *run)
{
    if (run == NULL) {
        return;
    }
    m3_runtime_workspace_dispose(&run->workspace);
    m3_command_executor_dispose(&run->executor);
    (void)memset(run, 0, sizeof(*run));
}
