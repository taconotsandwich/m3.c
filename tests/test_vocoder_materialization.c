/* SPDX-License-Identifier: GPL-2.0-only */

#include "vocoder_runtime_test.h"

#include "m3_test.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t calls;
    size_t total;
    size_t cancel_at;
    bool ordered;
} m3_vocoder_test_progress;

static bool m3_vocoder_test_progress_call(void *context,
                                           uint64_t completed,
                                           uint64_t total)
{
    m3_vocoder_test_progress *progress = context;

    if (completed != (uint64_t)progress->calls ||
        total != (uint64_t)progress->total) {
        progress->ordered = false;
    }
    ++progress->calls;
    return completed != (uint64_t)progress->cancel_at;
}

static uint32_t m3_vocoder_test_bits(float value)
{
    uint32_t bits;

    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float m3_vocoder_test_from_bits(uint32_t bits)
{
    float value;

    (void)memcpy(&value, &bits, sizeof(value));
    return value;
}

#pragma STDC FP_CONTRACT OFF
static float m3_vocoder_test_norm_value(float value, float gain, float norm)
{
    volatile float scale = gain / norm;
    volatile float result = value * scale;

    return result;
}
#pragma STDC FP_CONTRACT DEFAULT

static bool m3_vocoder_test_create_host_fixture(
    m3_vocoder_test_fixture *fixture, m3_error *error)
{
    m3_backend *backend = NULL;

    (void)memset(fixture, 0, sizeof(*fixture));
    return m3_backend_create_host(&backend, error) == M3_STATUS_OK &&
           m3_vocoder_test_fixture_create(
               fixture, backend, true, error);
}

static bool m3_vocoder_test_set_norm(
    m3_vocoder_test_fixture *fixture, const char *root,
    const float *gains, size_t gain_count, const float *values,
    size_t value_count, m3_error *error)
{
    char gain_name[M3_VOCODER_NAME_CAPACITY];
    char value_name[M3_VOCODER_NAME_CAPACITY];

    if (snprintf(gain_name, sizeof(gain_name), "%s.weight_g", root) < 0 ||
        snprintf(value_name, sizeof(value_name), "%s.weight_v", root) < 0) {
        return false;
    }
    return m3_vocoder_test_write_values(
               m3_vocoder_test_source(fixture, gain_name), gains,
               gain_count, error) &&
           m3_vocoder_test_write_values(
               m3_vocoder_test_source(fixture, value_name), values,
               value_count, error);
}

void test_vocoder_host_materialization(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    const m3_vocoder_weights *weights;
    m3_vocoder_test_progress progress;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    m3_error error;
    float direct[6] = {1.0F, -0.0F, 2.5F, -3.0F, 4.0F, 0.125F};
    float copied[6] = {0};

    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_create_host_fixture(&fixture, &error),
                   "create reduced host vocoder fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_vocoder_test_write_values(
            m3_vocoder_test_source(&fixture, "dec_in_proj.weight"),
            direct, 6U, &error),
        "write exact direct-copy fixture");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &before, &error) == M3_STATUS_OK,
                   "read source allocation statistics");
    m3_vocoder_materialize_io_init(&io);
    progress.calls = 0U;
    progress.total = fixture.plan.source_count;
    progress.cancel_at = SIZE_MAX;
    progress.ordered = true;
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       m3_vocoder_test_progress_call, &progress,
                       &error) == M3_STATUS_OK,
                   "materialize reduced host vocoder runtime");
    if (runtime == NULL) {
        m3_vocoder_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(test,
                   progress.ordered &&
                       progress.calls == fixture.plan.source_count + 1U,
                   "progress starts at zero and checkpoints each source");
    M3_TEST_EXPECT(test, runtime != NULL &&
                             runtime->backend == fixture.backend &&
                             runtime->weights.backend == fixture.backend &&
                             runtime->weights.count == fixture.plan.entry_count,
                   "runtime owns exact same-backend destination tensors");
    M3_TEST_EXPECT(test,
                   runtime->config.latent_channels == 4U &&
                       runtime->config.maximum_latent_length == 8U &&
                       runtime->config.decoder_input_channels == 2U &&
                       runtime->config.decoder_output_channels == 3U &&
                       runtime->config.initial_channels == 4U &&
                       runtime->config.block_count == 1U &&
                       runtime->config.residual_count == 1U &&
                       runtime->config.strides[0] == 2U,
                   "materialized runtime retains the complete reduced config");
    M3_TEST_EXPECT(test,
                   runtime != NULL &&
                       runtime->weights.allocated_bytes ==
                           m3_vocoder_test_runtime_bytes(&fixture),
                   "runtime live bytes equal the 91-to-121 plan reduction");
    weights = m3_vocoder_runtime_weights(runtime);
    M3_TEST_EXPECT(test,
                   weights != NULL && weights->block_count == 1U &&
                       weights->residual_count == 1U &&
                       weights->decoder_input_weight ==
                           &runtime->weights.views[0] &&
                       weights->convolution_input_weight ==
                           &runtime->weights.views[2] &&
                       weights->blocks[0].snake_alpha ==
                           &runtime->weights.views[4] &&
                       weights->blocks[0].residuals[0].conv2_bias ==
                           &runtime->weights.views[12] &&
                       weights->convolution_output_bias ==
                           &runtime->weights.views[15],
                   "typed binding is the sole complete workspace traversal");
    M3_TEST_EXPECT(
        test,
        weights != NULL && m3_vocoder_test_read_values(
                               weights->decoder_input_weight, copied, 6U,
                               &error),
        "read direct-copy destination");
    M3_TEST_EXPECT(test, memcmp(direct, copied, sizeof(direct)) == 0,
                   "direct F32 tensor copy preserves every bit");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       fixture.backend, &after, &error) == M3_STATUS_OK &&
                       after.live_allocated_bytes ==
                           before.live_allocated_bytes +
                               m3_vocoder_test_runtime_bytes(&fixture) &&
                       after.live_storage_count ==
                           before.live_storage_count +
                               fixture.plan.entry_count &&
                       after.peak_allocated_bytes ==
                           after.live_allocated_bytes &&
                       after.peak_storage_count == after.live_storage_count,
                   "live and peak backend ownership include source plus runtime");
    m3_vocoder_runtime_free(runtime);
    m3_vocoder_test_fixture_dispose(&fixture);
}

void test_vocoder_dim0_normalization(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    const m3_vocoder_weights *weights;
    m3_error error;
    float regular_g[] = {5.0F, 6.0F, 7.0F, 8.0F};
    float regular_v[84] = {0};
    float transpose_g[] = {2.0F, 3.0F, 4.0F, 5.0F};
    float transpose_v[32] = {0};
    float regular_out[84];
    float transpose_out[32];
    size_t row;

    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_create_host_fixture(&fixture, &error),
                   "create normalization fixture");
    if (fixture.backend == NULL) {
        return;
    }
    for (row = 0U; row < 4U; ++row) {
        regular_v[row * 21U] = 3.0F;
        regular_v[row * 21U + 1U] = 4.0F;
        transpose_v[row * 8U] = (float)(row + 1U);
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_set_norm(
                       &fixture, "conv_in", regular_g, 4U, regular_v,
                       84U, &error),
                   "set regular convolution norm source");
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_set_norm(
                       &fixture, "blocks.0.conv_t1", transpose_g, 4U,
                       transpose_v, 32U, &error),
                   "set transpose convolution norm source");
    m3_vocoder_materialize_io_init(&io);
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_OK,
                   "materialize dim-zero fixture");
    if (runtime == NULL) {
        m3_vocoder_test_fixture_dispose(&fixture);
        return;
    }
    weights = m3_vocoder_runtime_weights(runtime);
    M3_TEST_EXPECT(
        test,
        weights != NULL &&
            m3_vocoder_test_read_values(
                weights->convolution_input_weight, regular_out, 84U,
                &error) &&
            m3_vocoder_test_read_values(
                weights->blocks[0].transpose_weight, transpose_out, 32U,
                &error),
        "read both normalized convolution layouts");
    for (row = 0U; row < 4U; ++row) {
        M3_TEST_EXPECT_F32(test, regular_out[row * 21U],
                           m3_vocoder_test_norm_value(
                               3.0F, regular_g[row], 5.0F),
                           0.0F, 0.0F,
                           "regular convolution normalizes each dim-zero row");
        M3_TEST_EXPECT_F32(test, regular_out[row * 21U + 1U],
                           m3_vocoder_test_norm_value(
                               4.0F, regular_g[row], 5.0F),
                           0.0F, 0.0F,
                           "regular convolution keeps last-axis order");
        M3_TEST_EXPECT_F32(test, transpose_out[row * 8U],
                           transpose_g[row], 0.0F, 0.0F,
                           "transpose convolution normalizes dim zero");
    }
    m3_vocoder_runtime_free(runtime);
    m3_vocoder_test_fixture_dispose(&fixture);
}

void test_vocoder_fma_sensitive_order(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    const m3_vocoder_weights *weights;
    m3_error error;
    float gain[] = {1.0F};
    float values[14] = {0};
    float output[14] = {0};

    values[0] = m3_vocoder_test_from_bits(0x429f50ceU);
    values[1] = m3_vocoder_test_from_bits(0x3ea09507U);
    values[2] = m3_vocoder_test_from_bits(0x42f9a861U);
    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_create_host_fixture(&fixture, &error),
                   "create FMA-sensitive fixture");
    if (fixture.backend == NULL) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_set_norm(
                       &fixture, "conv_out", gain, 1U, values, 14U,
                       &error),
                   "set FMA-sensitive row");
    m3_vocoder_materialize_io_init(&io);
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, NULL) == M3_STATUS_OK,
                   "materialize with null error sink");
    if (runtime == NULL) {
        m3_vocoder_test_fixture_dispose(&fixture);
        return;
    }
    weights = m3_vocoder_runtime_weights(runtime);
    M3_TEST_EXPECT(test,
                   weights != NULL &&
                       m3_vocoder_test_read_values(
                           weights->convolution_output_weight, output,
                           14U, &error),
                   "read FMA-sensitive result");
    M3_TEST_EXPECT(test, m3_vocoder_test_bits(output[0]) == 0x3f09b645U,
                   "separate F32 product-add matches fixed row oracle");
    M3_TEST_EXPECT(test, m3_vocoder_test_bits(output[0]) != 0x3f09b644U,
                   "row oracle rejects fused accumulation result");
    m3_vocoder_runtime_free(runtime);
    m3_vocoder_test_fixture_dispose(&fixture);
}
