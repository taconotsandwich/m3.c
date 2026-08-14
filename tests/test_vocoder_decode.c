/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "vocoder_decode_test.h"
#include "vocoder_runtime_test.h"
#include "weight_stage_fixture.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t counts[7U];
    size_t call_count;
    uint64_t cancel_at;
    bool totals_valid;
} m3_vocoder_decode_progress;

static bool m3_vocoder_decode_progress_call(
    void *context, uint64_t completed, uint64_t total)
{
    m3_vocoder_decode_progress *progress = context;

    if (progress->call_count < 7U) {
        progress->counts[progress->call_count] = completed;
    }
    ++progress->call_count;
    progress->totals_valid = progress->totals_valid && total == 73U;
    return completed != progress->cancel_at;
}

static void m3_vocoder_decode_progress_init(
    m3_vocoder_decode_progress *progress, uint64_t cancel_at)
{
    (void)memset(progress, 0, sizeof(*progress));
    progress->cancel_at = cancel_at;
    progress->totals_valid = true;
}

static bool m3_vocoder_decode_progress_complete(
    const m3_vocoder_decode_progress *progress)
{
    static const uint64_t expected[] = {0U, 2U, 19U, 36U,
                                        53U, 70U, 73U};

    return progress->totals_valid && progress->call_count == 7U &&
           memcmp(progress->counts, expected, sizeof(expected)) == 0;
}

static float m3_vocoder_decode_from_bits(uint32_t bits)
{
    float value;

    (void)memcpy(&value, &bits, sizeof(value));
    return value;
}

void test_vocoder_decode_plan_oracles(m3_test_context *test)
{
    m3_vocoder_plan_config official;
    m3_vocoder_plan_config reduced;
    m3_vocoder_decode_measurement measurement = {0};
    m3_vocoder_decode_measurement sentinel;
    m3_error error;
    size_t slot;

    m3_vocoder_plan_official_config(&official);
    M3_TEST_EXPECT(test,
                   official.latent_channels == 128U &&
                       official.maximum_latent_length == 689U &&
                       official.decoder_input_channels == 64U,
                   "official decoder config locks folded latent channels");
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_measure(
                       &official, 1U, &measurement, &error) ==
                           M3_STATUS_OK &&
                       measurement.output_length == 512U &&
                       measurement.output_bytes == 4096U &&
                       measurement.workspace_bytes == 1179648U,
                   "official one-frame decoder memory plan is exact");
    for (slot = 0U; slot < M3_VOCODER_DECODE_BUFFER_COUNT; ++slot) {
        M3_TEST_EXPECT(test, measurement.buffer_bytes[slot] == 393216U,
                       "official rotation gives each buffer an exact peak");
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_measure(
                       &official, 689U, &measurement, &error) ==
                           M3_STATUS_OK &&
                       measurement.output_length == 352768U &&
                       measurement.output_bytes == 2822144U &&
                       measurement.workspace_bytes == 812777472U,
                   "official maximum chunk memory plan is exact");
    for (slot = 0U; slot < M3_VOCODER_DECODE_BUFFER_COUNT; ++slot) {
        M3_TEST_EXPECT(test,
                       measurement.buffer_bytes[slot] == 270925824U,
                       "official maximum chunk locks all buffer peaks");
    }
    sentinel = measurement;
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_measure(
                       &official, 690U, &measurement, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       memcmp(&measurement, &sentinel, sizeof(sentinel)) ==
                           0,
                   "official config owns the atomic latent length bound");
    m3_vocoder_decode_test_config(&reduced);
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_measure(
                       &reduced, 2U, &measurement, &error) ==
                           M3_STATUS_OK &&
                       measurement.output_length == 32U &&
                       measurement.output_bytes == 256U &&
                       measurement.workspace_bytes == 768U &&
                       measurement.buffer_bytes[0] == 256U &&
                       measurement.buffer_bytes[1] == 256U &&
                       measurement.buffer_bytes[2] == 256U,
                   "reduced full topology uses the same measured rotation");
    reduced.latent_channels = 4U;
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_measure(
                       &reduced, 2U, &measurement, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "decoder config rejects an unfolded channel mismatch");
}

void test_vocoder_decode_host_oracle(m3_test_context *test)
{
    static const uint32_t expected_bits[64U] = {
        0x3d6ae6e7U, 0x3cc0a002U, 0x3d6a0ddbU, 0x3ccb59a2U,
        0x3cf71eedU, 0x3c2cc4f0U, 0x3d3a0252U, 0x3d127f82U,
        0x3c960d5eU, 0x3b8f6a6eU, 0x3d1c8245U, 0x3d07ea8fU,
        0x3c898427U, 0x3c1bc0d9U, 0x3d505a77U, 0x3d31040eU,
        0x3ba13393U, 0xbc336b4dU, 0x3c9d18b0U, 0x3cd01d95U,
        0xbb3af2dfU, 0x3aaffda9U, 0x3d253b4dU, 0x3d1beef7U,
        0xbc707768U, 0xbcc62f5aU, 0xbc26f687U, 0x3acccb9dU,
        0xbd01c05cU, 0xb9c2d229U, 0x3bb0c25fU, 0x3ba56a3fU,
        0xbdf1da88U, 0xbd0a291aU, 0xbdfee511U, 0xbdc45204U,
        0xbd90a8d6U, 0x3cec4738U, 0xbdacfd65U, 0xbddbbe37U,
        0xbd11f9b5U, 0x3d4f3d7aU, 0xbdcae7b5U, 0xbe0fc106U,
        0xbceb66f0U, 0x3d90bfdbU, 0xbda8f5ecU, 0xbdfead08U,
        0x3cc0b787U, 0x3dce2c29U, 0xbd17098cU, 0xbdb71a6aU,
        0x3d43ff0bU, 0x3d57cf63U, 0xbd82a74bU, 0xbdbd8cc3U,
        0x3d49796eU, 0x3d63159aU, 0x3c0e70a7U, 0xbc5632edU,
        0x3d7a3b86U, 0x3c23a12aU, 0xbc7342e6U, 0xbcb0780cU
    };
    const float input_values[] = {0.125F, -0.25F, 0.75F, -0.5F};
    m3_vocoder_decode_progress progress;
    m3_vocoder_runtime *runtime = NULL;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *input_storage = NULL;
    m3_tensor_view latents;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    float actual[64U];
    m3_error error;
    size_t index;

    m3_vocoder_output_init(&output);
    m3_vocoder_decode_progress_init(&progress, UINT64_MAX);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_vocoder_decode_test_runtime(
                           backend, &runtime, &error) &&
                       m3_vocoder_decode_test_latents(
                           backend, 2U, input_values, &input_storage,
                           &latents, &error) &&
                       m3_backend_get_allocation_stats(
                           backend, &before, &error) == M3_STATUS_OK,
                   "create reduced host decoder after staging lifetime ends");
    if (runtime == NULL || input_storage == NULL) {
        m3_backend_free(backend);
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       runtime, &latents, m3_vocoder_decode_progress_call,
                       &progress, &output, &error) == M3_STATUS_OK &&
                       m3_vocoder_decode_progress_complete(&progress),
                   "decode the exact 73-op host topology in seven checkpoints");
    M3_TEST_EXPECT(test,
                   output.storage != NULL &&
                       output.waveform.storage == output.storage &&
                       output.waveform.metadata.dtype == M3_DTYPE_F32 &&
                       output.waveform.metadata.rank == 3U &&
                       output.waveform.metadata.shape[0] == 1U &&
                       output.waveform.metadata.shape[1] == 2U &&
                       output.waveform.metadata.shape[2] == 32U &&
                       m3_tensor_is_contiguous(&output.waveform) &&
                       m3_storage_read(output.storage, 0U, actual,
                                       sizeof(actual), &error) == M3_STATUS_OK,
                   "publish one owned contiguous planar stereo waveform");
    for (index = 0U; index < 64U; ++index) {
        M3_TEST_EXPECT_F32(
            test, actual[index],
            m3_vocoder_decode_from_bits(expected_bits[index]),
            0x1p-19F, 0x1p-19F,
            "host decoder matches the independent reduced oracle");
    }
    M3_TEST_EXPECT(test,
                   memcmp(actual, &actual[32U], 32U * sizeof(float)) != 0,
                   "fold [1,2*C,L] into two independent [2,C,L] streams");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       backend, &after, &error) == M3_STATUS_OK &&
                       after.live_allocated_bytes ==
                           before.live_allocated_bytes + sizeof(actual) &&
                       after.live_storage_count == before.live_storage_count +
                                                        1U,
                   "release all three private work buffers after publication");
    m3_vocoder_runtime_free(runtime);
    runtime = NULL;
    M3_TEST_EXPECT(test,
                   m3_storage_read(output.storage, 0U, actual,
                                   sizeof(actual), &error) == M3_STATUS_OK,
                   "published waveform outlives its materialized runtime");
    m3_vocoder_output_dispose(&output);
    m3_storage_free(input_storage);
    m3_backend_free(backend);
}

void test_vocoder_decode_atomic_cancel(m3_test_context *test)
{
    const float input_values[] = {0.125F, -0.25F, 0.75F, -0.5F};
    const float sentinel_values[] = {-7.0F, 11.0F};
    m3_vocoder_decode_progress progress;
    m3_vocoder_runtime *runtime = NULL;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *input_storage = NULL;
    m3_storage *sentinel_storage = NULL;
    m3_tensor_view latents;
    m3_tensor_view strided;
    m3_tensor_view sentinel_view;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    float sentinel_after[2U] = {0};
    m3_error error;

    m3_vocoder_output_init(&output);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_vocoder_decode_test_runtime(
                           backend, &runtime, &error) &&
                       m3_vocoder_decode_test_latents(
                           backend, 2U, input_values, &input_storage,
                           &latents, &error) &&
                       m3_vocoder_decode_test_latents(
                           backend, 1U, sentinel_values, &sentinel_storage,
                           &sentinel_view, &error),
                   "create decoder cancellation ownership fixture");
    if (runtime == NULL || sentinel_storage == NULL) {
        m3_storage_free(input_storage);
        m3_backend_free(backend);
        return;
    }
    output.storage = sentinel_storage;
    output.waveform = sentinel_view;
    m3_vocoder_decode_progress_init(&progress, 36U);
    (void)m3_backend_get_allocation_stats(backend, &before, &error);
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       runtime, &latents, m3_vocoder_decode_progress_call,
                       &progress, &output, &error) == M3_STATUS_CANCELLED &&
                       progress.call_count == 4U && progress.counts[3] == 36U &&
                       output.storage == sentinel_storage &&
                       m3_storage_read(output.storage, 0U, sentinel_after,
                                       sizeof(sentinel_after), &error) ==
                           M3_STATUS_OK &&
                       memcmp(sentinel_after, sentinel_values,
                              sizeof(sentinel_values)) == 0 &&
                       m3_backend_get_allocation_stats(
                           backend, &after, &error) == M3_STATUS_OK &&
                       before.live_allocated_bytes ==
                           after.live_allocated_bytes &&
                       before.live_storage_count == after.live_storage_count,
                   "later-batch cancellation preserves output and live memory");
    m3_vocoder_decode_progress_init(&progress, 73U);
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       runtime, &latents, m3_vocoder_decode_progress_call,
                       &progress, &output, &error) == M3_STATUS_CANCELLED &&
                       progress.call_count == 7U &&
                       progress.counts[6] == 73U &&
                       output.storage == sentinel_storage &&
                       m3_storage_read(output.storage, 0U, sentinel_after,
                                       sizeof(sentinel_after), &error) ==
                           M3_STATUS_OK &&
                       memcmp(sentinel_after, sentinel_values,
                              sizeof(sentinel_values)) == 0,
                   "final checkpoint cancellation withholds the built waveform");
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       runtime, NULL, NULL, NULL, &output, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       output.storage == sentinel_storage,
                   "null error sink preserves output on validation failure");
    strided = latents;
    strided.byte_strides[1] = sizeof(float);
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       runtime, &strided, NULL, NULL, &output, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       output.storage == sentinel_storage,
                   "decoder rejects a strided latent with a null error sink");
    m3_vocoder_decode_progress_init(&progress, UINT64_MAX);
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       runtime, &latents, m3_vocoder_decode_progress_call,
                       &progress, &output, &error) == M3_STATUS_OK &&
                       m3_vocoder_decode_progress_complete(&progress) &&
                       output.storage != sentinel_storage,
                   "retry atomically replaces the prior owned waveform");
    sentinel_storage = NULL;
    m3_vocoder_output_dispose(&output);
    m3_storage_free(input_storage);
    m3_vocoder_runtime_free(runtime);
    m3_backend_free(backend);
}

void test_vocoder_decode_rejects_malformed_input(m3_test_context *test)
{
    const float input_values[] = {0.125F, -0.25F, 0.75F, -0.5F};
    const float sentinel_values[] = {-7.0F, 11.0F};
    m3_weight_stage_fake_context *context = NULL;
    m3_vocoder_decode_progress progress;
    m3_vocoder_runtime *runtime = NULL;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *input_storage = NULL;
    m3_storage *sentinel_storage = NULL;
    m3_storage *published_storage;
    m3_tensor_view latents;
    m3_tensor_view malformed;
    m3_tensor_view sentinel_view;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    float sentinel_after[2U] = {0};
    m3_error error;
    size_t allocation_calls;
    size_t free_calls;
    bool created;

    m3_vocoder_output_init(&output);
    created = m3_weight_stage_test_fake_backend_create(
        (uint64_t)SIZE_MAX, 0U, 0U, &backend, &context, &error) &&
        m3_vocoder_decode_test_runtime(backend, &runtime, &error) &&
        m3_vocoder_decode_test_latents(
            backend, 2U, input_values, &input_storage, &latents, &error) &&
        m3_vocoder_decode_test_latents(
            backend, 1U, sentinel_values, &sentinel_storage,
            &sentinel_view, &error);
    M3_TEST_EXPECT(test, created,
                   "create malformed decoder input fixture");
    if (!created) {
        m3_storage_free(sentinel_storage);
        m3_storage_free(input_storage);
        m3_vocoder_runtime_free(runtime);
        m3_backend_free(backend);
        return;
    }
    published_storage = sentinel_storage;
    output.storage = published_storage;
    output.waveform = sentinel_view;
    sentinel_storage = NULL;
    malformed = latents;
    ++malformed.metadata.element_count;
    allocation_calls = context->allocation_calls;
    free_calls = context->free_calls;
    m3_vocoder_decode_progress_init(&progress, UINT64_MAX);
    M3_TEST_EXPECT(test,
                   m3_tensor_is_contiguous(&malformed) &&
                       m3_backend_get_allocation_stats(
                           backend, &before, &error) == M3_STATUS_OK &&
                       m3_vocoder_decode_chunk(
                           runtime, &malformed,
                           m3_vocoder_decode_progress_call, &progress,
                           &output, &error) == M3_STATUS_INVALID_ARGUMENT &&
                       progress.call_count == 0U &&
                       context->allocation_calls == allocation_calls &&
                       context->free_calls == free_calls &&
                       output.storage == published_storage &&
                       memcmp(&output.waveform, &sentinel_view,
                              sizeof(sentinel_view)) == 0 &&
                       m3_backend_get_allocation_stats(
                           backend, &after, &error) == M3_STATUS_OK &&
                       before.live_allocated_bytes ==
                           after.live_allocated_bytes &&
                       before.live_storage_count == after.live_storage_count,
                   "reject inconsistent contiguous metadata before allocation");
    malformed = latents;
    malformed.byte_offset = sizeof(float);
    allocation_calls = context->allocation_calls;
    free_calls = context->free_calls;
    m3_vocoder_decode_progress_init(&progress, UINT64_MAX);
    M3_TEST_EXPECT(test,
                   m3_tensor_is_contiguous(&malformed) &&
                       m3_backend_get_allocation_stats(
                           backend, &before, &error) == M3_STATUS_OK &&
                       m3_vocoder_decode_chunk(
                           runtime, &malformed,
                           m3_vocoder_decode_progress_call, &progress,
                           &output, NULL) == M3_STATUS_OUT_OF_RANGE &&
                       progress.call_count == 0U &&
                       context->allocation_calls == allocation_calls &&
                       context->free_calls == free_calls &&
                       output.storage == published_storage &&
                       memcmp(&output.waveform, &sentinel_view,
                              sizeof(sentinel_view)) == 0 &&
                       m3_backend_get_allocation_stats(
                           backend, &after, &error) == M3_STATUS_OK &&
                       before.live_allocated_bytes ==
                           after.live_allocated_bytes &&
                       before.live_storage_count == after.live_storage_count,
                   "reject out-of-bounds contiguous view before allocation");
    M3_TEST_EXPECT(test,
                   m3_storage_read(
                       published_storage, 0U, sentinel_after,
                       sizeof(sentinel_after), &error) == M3_STATUS_OK &&
                       memcmp(sentinel_after, sentinel_values,
                              sizeof(sentinel_values)) == 0,
                   "malformed inputs preserve the published waveform bytes");
    m3_vocoder_output_dispose(&output);
    m3_storage_free(input_storage);
    m3_vocoder_runtime_free(runtime);
    m3_backend_free(backend);
    (void)context;
}

static void m3_vocoder_decode_limit_case(
    m3_test_context *test, uint64_t maximum_storage,
    uint64_t recommended, m3_status expected, const char *description)
{
    m3_weight_stage_fake_context *context = NULL;
    m3_vocoder_decode_progress progress;
    m3_vocoder_plan_config config;
    m3_vocoder_runtime *runtime = NULL;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *input_storage = NULL;
    m3_tensor_view latents;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    float *values = calloc(1378U, sizeof(*values));
    m3_error error;
    bool created;

    m3_vocoder_output_init(&output);
    m3_vocoder_decode_test_config(&config);
    config.maximum_latent_length = 689U;
    created = values != NULL && m3_weight_stage_test_fake_backend_create(
        maximum_storage, recommended, 0U, &backend, &context, &error) &&
        m3_vocoder_decode_test_runtime_config(
            backend, &config, &runtime, &error) &&
        m3_vocoder_decode_test_latents(
            backend, 689U, values, &input_storage, &latents, &error);
    M3_TEST_EXPECT(test, created, "create decoder preflight limit fixture");
    if (created) {
        m3_vocoder_decode_progress_init(&progress, UINT64_MAX);
        (void)m3_backend_get_allocation_stats(backend, &before, &error);
        M3_TEST_EXPECT(test,
                       m3_vocoder_decode_chunk(
                           runtime, &latents,
                           m3_vocoder_decode_progress_call, &progress,
                           &output, &error) == expected &&
                           progress.call_count == 0U &&
                           output.storage == NULL &&
                           m3_backend_get_allocation_stats(
                               backend, &after, &error) == M3_STATUS_OK &&
                           before.live_allocated_bytes ==
                               after.live_allocated_bytes &&
                           before.live_storage_count ==
                               after.live_storage_count,
                       description);
    }
    free(values);
    m3_storage_free(input_storage);
    m3_vocoder_runtime_free(runtime);
    m3_backend_free(backend);
    (void)context;
}

void test_vocoder_decode_preflight_limits(m3_test_context *test)
{
    m3_vocoder_decode_limit_case(
        test, 8000U, 0U, M3_STATUS_OUT_OF_RANGE,
        "reject one planned buffer above backend storage limit");
    m3_vocoder_decode_limit_case(
        test, (uint64_t)SIZE_MAX, 200000U, M3_STATUS_OUT_OF_MEMORY,
        "reject live plus workspace plus output above working set");
}

void test_vocoder_decode_rejects_partial_topology(m3_test_context *test)
{
    const uint64_t shape[] = {1U, 4U, 1U};
    const float values[4U] = {0};
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *storage = NULL;
    m3_tensor_view latents;
    m3_error error;

    m3_vocoder_output_init(&output);
    m3_vocoder_materialize_io_init(&io);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_vocoder_test_fixture_create(
                           &fixture, backend, false, &error) &&
                       m3_vocoder_runtime_create_core(
                           &runtime, &fixture.stage, &fixture.plan, &io,
                           NULL, NULL, &error) == M3_STATUS_OK,
                   "construct the existing one-block materialization runtime");
    m3_vocoder_test_fixture_dispose(&fixture);
    if (runtime == NULL) {
        m3_backend_free(backend);
        return;
    }
    m3_tensor_view_init(&latents);
    M3_TEST_EXPECT(test,
                   m3_storage_allocate(
                       backend, sizeof(values), 64U, &storage, &error) ==
                           M3_STATUS_OK &&
                       m3_tensor_view_contiguous(
                           &latents, storage, M3_DTYPE_F32, 3U, shape, 0U,
                           &error) == M3_STATUS_OK &&
                       m3_storage_write(
                           storage, 0U, values, sizeof(values), &error) ==
                           M3_STATUS_OK,
                   "create partial-topology latent input");
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       runtime, &latents, NULL, NULL, &output, NULL) ==
                           M3_STATUS_INVALID_ARGUMENT &&
                       output.storage == NULL,
                   "single decoder rejects a non-4x3 materialization runtime");
    m3_storage_free(storage);
    m3_vocoder_runtime_free(runtime);
    m3_backend_free(backend);
}
