/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "music3_engine_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    m3_music3_test_corruption corruption;
    uint64_t frames;
    m3_music3_phase phase;
    uint64_t total;
} m3_music3_malformed_case;

static bool m3_music3_stats_equal(
    const m3_backend_allocation_stats *left,
    const m3_backend_allocation_stats *right)
{
    return left->live_allocated_bytes == right->live_allocated_bytes &&
           left->live_storage_count == right->live_storage_count;
}

static bool m3_music3_rng_equal(const m3_rng *left, const m3_rng *right)
{
    return left->state == right->state &&
           left->increment == right->increment &&
           left->spare_normal == right->spare_normal &&
           left->has_spare_normal == right->has_spare_normal;
}

static bool m3_music3_request_equal(
    const m3_music3_request *left, const m3_music3_request *right)
{
    return left->caption.data == right->caption.data &&
           left->caption.length == right->caption.length &&
           left->lyrics.data == right->lyrics.data &&
           left->lyrics.length == right->lyrics.length &&
           left->maximum_frames == right->maximum_frames &&
           left->seed == right->seed && left->sequence == right->sequence;
}

static bool m3_music3_output_info_equal(
    const m3_music3_output_info *left,
    const m3_music3_output_info *right)
{
    return left->sample_rate == right->sample_rate &&
           left->channel_count == right->channel_count &&
           left->samples_per_channel == right->samples_per_channel;
}

static void m3_music3_test_runtime_duplicate_validator(
    m3_test_context *test)
{
    m3_backend *backend = NULL;
    m3_vocoder_runtime *runtime = NULL;
    m3_backend_allocation_stats before = {0U, 0U, 0U, 0U};
    m3_backend_allocation_stats after = {0U, 0U, 0U, 0U};
    m3_error error;
    m3_status status = m3_backend_create_host(&backend, &error);
    bool ready = status == M3_STATUS_OK;
    bool duplicate_message = false;

    if (ready) {
        ready = m3_backend_get_allocation_stats(
                    backend, &before, &error) == M3_STATUS_OK;
    }
    runtime = ready ? calloc(1U, sizeof(*runtime)) : NULL;
    ready = ready && runtime != NULL;
    if (ready) {
        runtime->backend = backend;
        m3_vocoder_plan_official_config(&runtime->config);
        m3_runtime_workspace_init(&runtime->weights);
        runtime->weights.backend = backend;
        runtime->weights.count = M3_VOCODER_RUNTIME_WEIGHT_COUNT;
        runtime->weights.storages = calloc(
            runtime->weights.count, sizeof(*runtime->weights.storages));
        runtime->weights.views = calloc(
            runtime->weights.count, sizeof(*runtime->weights.views));
        ready = runtime->weights.storages != NULL &&
                runtime->weights.views != NULL;
    }
    if (ready) {
        ready = m3_storage_allocate(
                    backend, sizeof(float), 64U,
                    &runtime->weights.storages[0], &error) ==
                M3_STATUS_OK;
    }
    if (ready) {
        runtime->weights.storages[1] = runtime->weights.storages[0];
        status = m3_vocoder_runtime_validate(runtime, backend, &error);
        duplicate_message = strcmp(
            error.message,
            "published vocoder tensors share owned storage") == 0;
    }
    m3_vocoder_runtime_free(runtime);
    runtime = NULL;
    if (ready) {
        ready = m3_backend_get_allocation_stats(
                    backend, &after, &error) == M3_STATUS_OK &&
                m3_music3_stats_equal(&before, &after);
    }
    M3_TEST_EXPECT(
        test,
        ready && status == M3_STATUS_INVALID_FORMAT &&
            duplicate_message,
        "shared vocoder validator rejects duplicate owner and disposer frees once");
    m3_backend_free(backend);
}

void m3_test_music3_result_validation(m3_test_context *test)
{
    static const m3_music3_malformed_case cases[] = {
        {M3_MUSIC3_TEST_CORRUPT_STAGE_EMPTY, 1U,
         M3_MUSIC3_PHASE_STAGE_SEMANTIC,
         M3_MUSIC3_STAGE_SEMANTIC_BYTES},
        {M3_MUSIC3_TEST_CORRUPT_STAGE_VIEW, 1U,
         M3_MUSIC3_PHASE_STAGE_SEMANTIC,
         M3_MUSIC3_STAGE_SEMANTIC_BYTES},
        {M3_MUSIC3_TEST_CORRUPT_STAGE_DUPLICATE, 1U,
         M3_MUSIC3_PHASE_STAGE_SEMANTIC,
         M3_MUSIC3_STAGE_SEMANTIC_BYTES},
        {M3_MUSIC3_TEST_CORRUPT_STAGE_PROGRESS, 1U,
         M3_MUSIC3_PHASE_STAGE_SEMANTIC,
         M3_MUSIC3_STAGE_SEMANTIC_BYTES},
        {M3_MUSIC3_TEST_CORRUPT_SEMANTIC_LIMIT, 1U,
         M3_MUSIC3_PHASE_SEMANTIC, 97U},
        {M3_MUSIC3_TEST_CORRUPT_SEMANTIC_OVERSIZE, 1U,
         M3_MUSIC3_PHASE_SEMANTIC, 97U},
        {M3_MUSIC3_TEST_CORRUPT_FLOW_LENGTH, 1U,
         M3_MUSIC3_PHASE_FLOW, 30U},
        {M3_MUSIC3_TEST_CORRUPT_FLOW_DUPLICATE, 300U,
         M3_MUSIC3_PHASE_FLOW, 60U},
        {M3_MUSIC3_TEST_CORRUPT_RUNTIME_VIEW, 1U,
         M3_MUSIC3_PHASE_MATERIALIZE_VOCODER,
         M3_VOCODER_SOURCE_WEIGHT_COUNT},
        {M3_MUSIC3_TEST_CORRUPT_RUNTIME_DUPLICATE, 1U,
         M3_MUSIC3_PHASE_MATERIALIZE_VOCODER,
         M3_VOCODER_SOURCE_WEIGHT_COUNT},
        {M3_MUSIC3_TEST_CORRUPT_DECODE_LENGTH, 1U,
         M3_MUSIC3_PHASE_DECODE,
         M3_VOCODER_DECODE_OPERATION_COUNT},
        {M3_MUSIC3_TEST_CORRUPT_DECODE_DUPLICATE, 300U,
         M3_MUSIC3_PHASE_DECODE,
         2U * M3_VOCODER_DECODE_OPERATION_COUNT},
        {M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_SHORT, 1U,
         M3_MUSIC3_PHASE_ASSEMBLE, 2U},
        {M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_LONG, 1U,
         M3_MUSIC3_PHASE_ASSEMBLE, 2U},
        {M3_MUSIC3_TEST_CORRUPT_ASSEMBLE_OVERSIZE, 1U,
         M3_MUSIC3_PHASE_ASSEMBLE, 2U},
    };
    size_t index;

    for (index = 0U; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        const m3_music3_malformed_case *item = &cases[index];
        m3_music3_test_fixture fixture;
        m3_music3_test_progress progress;
        m3_music3_request request;
        m3_music3_output *output = NULL;
        m3_backend_allocation_stats before = {0U, 0U, 0U, 0U};
        m3_backend_allocation_stats after = {0U, 0U, 0U, 0U};
        m3_error error;
        char message[256];
        m3_status status = m3_music3_test_fixture_create(
            &fixture, false, &error);
        bool valid;

        M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                       "create malformed-result fixture");
        if (status != M3_STATUS_OK) {
            continue;
        }
        fixture.corruption = item->corruption;
        fixture.scripted_frames = item->frames;
        m3_music3_test_request(&request, item->frames);
        m3_music3_test_progress_init(&progress);
        (void)m3_backend_get_allocation_stats(
            fixture.engine->backend, &before, &error);
        status = m3_music3_generate(
            fixture.engine, &request, m3_music3_test_progress_call,
            &progress, &output, &error);
        (void)m3_backend_get_allocation_stats(
            fixture.engine->backend, &after, &error);
        valid = status == M3_STATUS_INTERNAL && output == NULL &&
                progress.contract_valid &&
                !m3_music3_test_phase_reached(
                    &progress, item->phase, item->total) &&
                (item->phase == M3_MUSIC3_PHASE_ASSEMBLE ||
                 !m3_music3_test_phase_reached(
                     &progress,
                     (m3_music3_phase)((int)item->phase + 1), 0U)) &&
                m3_music3_stats_equal(&before, &after);
        (void)snprintf(
            message, sizeof(message),
            "malformed child %d: status %d output %d contract %d final %d next %d stats %d: %s",
            (int)item->corruption, (int)status, output != NULL,
            progress.contract_valid,
            m3_music3_test_phase_reached(
                &progress, item->phase, item->total),
            item->phase == M3_MUSIC3_PHASE_ASSEMBLE
                ? 0
                : m3_music3_test_phase_reached(
                      &progress,
                      (m3_music3_phase)((int)item->phase + 1), 0U),
            m3_music3_stats_equal(&before, &after), error.message);
        M3_TEST_EXPECT(test, valid, message);
        m3_music3_output_free(output);
        m3_music3_test_fixture_dispose(&fixture);
    }
    m3_music3_test_runtime_duplicate_validator(test);
}

typedef struct {
    m3_music3_phase phase;
    uint64_t completed;
    uint64_t frames;
} m3_music3_cancel_case;

void m3_test_music3_cancellation_atomicity(m3_test_context *test)
{
    static const m3_music3_cancel_case cases[] = {
        {M3_MUSIC3_PHASE_PREPARE, 0U, 1U},
        {M3_MUSIC3_PHASE_PREPARE, 1U, 1U},
        {M3_MUSIC3_PHASE_STAGE_SEMANTIC, 0U, 1U},
        {M3_MUSIC3_PHASE_STAGE_SEMANTIC, 1U, 1U},
        {M3_MUSIC3_PHASE_STAGE_SEMANTIC, UINT64_C(17168951296), 1U},
        {M3_MUSIC3_PHASE_STAGE_SEMANTIC, UINT64_C(17168951297), 1U},
        {M3_MUSIC3_PHASE_STAGE_SEMANTIC,
         M3_MUSIC3_STAGE_SEMANTIC_BYTES, 1U},
        {M3_MUSIC3_PHASE_SEMANTIC, 0U, 1U},
        {M3_MUSIC3_PHASE_SEMANTIC, 1U, 1U},
        {M3_MUSIC3_PHASE_SEMANTIC, 97U, 1U},
        {M3_MUSIC3_PHASE_STAGE_FLOW, 0U, 1U},
        {M3_MUSIC3_PHASE_STAGE_FLOW, 1U, 1U},
        {M3_MUSIC3_PHASE_STAGE_FLOW, UINT64_C(100671524), 1U},
        {M3_MUSIC3_PHASE_STAGE_FLOW, UINT64_C(100671525), 1U},
        {M3_MUSIC3_PHASE_STAGE_FLOW, M3_MUSIC3_STAGE_FLOW_BYTES, 1U},
        {M3_MUSIC3_PHASE_FLOW, 0U, 1U},
        {M3_MUSIC3_PHASE_FLOW, 1U, 1U},
        {M3_MUSIC3_PHASE_FLOW, 30U, 1U},
        {M3_MUSIC3_PHASE_FLOW, 30U, 300U},
        {M3_MUSIC3_PHASE_FLOW, 31U, 300U},
        {M3_MUSIC3_PHASE_FLOW, 60U, 300U},
        {M3_MUSIC3_PHASE_STAGE_VOCODER, 0U, 1U},
        {M3_MUSIC3_PHASE_STAGE_VOCODER, 1U, 1U},
        {M3_MUSIC3_PHASE_STAGE_VOCODER,
         M3_MUSIC3_STAGE_VOCODER_BYTES, 1U},
        {M3_MUSIC3_PHASE_MATERIALIZE_VOCODER, 0U, 1U},
        {M3_MUSIC3_PHASE_MATERIALIZE_VOCODER, 1U, 1U},
        {M3_MUSIC3_PHASE_MATERIALIZE_VOCODER,
         M3_VOCODER_SOURCE_WEIGHT_COUNT, 1U},
        {M3_MUSIC3_PHASE_DECODE, 0U, 1U},
        {M3_MUSIC3_PHASE_DECODE, 1U, 1U},
        {M3_MUSIC3_PHASE_DECODE,
         M3_VOCODER_DECODE_OPERATION_COUNT, 1U},
        {M3_MUSIC3_PHASE_DECODE,
         M3_VOCODER_DECODE_OPERATION_COUNT, 300U},
        {M3_MUSIC3_PHASE_DECODE,
         M3_VOCODER_DECODE_OPERATION_COUNT + 1U, 300U},
        {M3_MUSIC3_PHASE_DECODE,
         2U * M3_VOCODER_DECODE_OPERATION_COUNT, 300U},
        {M3_MUSIC3_PHASE_ASSEMBLE, 0U, 1U},
        {M3_MUSIC3_PHASE_ASSEMBLE, 1U, 1U},
        {M3_MUSIC3_PHASE_ASSEMBLE, 2U, 1U},
        {M3_MUSIC3_PHASE_ASSEMBLE, 2U, 300U},
        {M3_MUSIC3_PHASE_ASSEMBLE, 3U, 300U},
        {M3_MUSIC3_PHASE_ASSEMBLE, 4U, 300U},
    };
    m3_music3_test_fixture fixture;
    m3_music3_request request;
    m3_music3_request request_copy;
    m3_music3_output *output = NULL;
    m3_music3_output *preserved;
    m3_backend_allocation_stats baseline = {0U, 0U, 0U, 0U};
    m3_rng semantic_entry;
    m3_rng semantic_exit;
    m3_rng flow_entry;
    m3_rng flow_exit;
    m3_music3_output_info expected_info;
    m3_music3_output_info actual_info;
    float *expected_zero = NULL;
    float *expected_one = NULL;
    float *actual_zero = NULL;
    float *actual_one = NULL;
    size_t samples = 0U;
    size_t bytes = 0U;
    m3_error error;
    m3_status status = m3_music3_test_fixture_create(
        &fixture, false, &error);
    bool seeded = false;
    size_t index;

    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create cancellation fixture");
    if (status != M3_STATUS_OK) {
        return;
    }
    fixture.scripted_frames = 1U;
    m3_music3_test_request(&request, 1U);
    request_copy = request;
    status = m3_music3_generate(
        fixture.engine, &request, NULL, NULL, &output, &error);
    if (status == M3_STATUS_OK && output != NULL &&
        m3_music3_output_get_info(output, &expected_info, &error) ==
            M3_STATUS_OK &&
        expected_info.samples_per_channel <= SIZE_MAX / sizeof(float)) {
        samples = (size_t)expected_info.samples_per_channel;
        bytes = samples * sizeof(float);
        expected_zero = malloc(bytes);
        expected_one = malloc(bytes);
        actual_zero = malloc(bytes);
        actual_one = malloc(bytes);
    }
    seeded = status == M3_STATUS_OK && output != NULL &&
        fixture.semantic_rng_observed && fixture.flow_rng_observed &&
        expected_zero != NULL && expected_one != NULL &&
        actual_zero != NULL && actual_one != NULL &&
        m3_music3_output_read_channel_f32(
            output, 0U, 0U, expected_zero, samples, &error) ==
            M3_STATUS_OK &&
        m3_music3_output_read_channel_f32(
            output, 1U, 0U, expected_one, samples, &error) ==
            M3_STATUS_OK &&
        m3_backend_get_allocation_stats(
            fixture.engine->backend, &baseline, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, seeded,
                   "seed old output before cancellation matrix");
    if (!seeded) {
        free(actual_one);
        free(actual_zero);
        free(expected_one);
        free(expected_zero);
        m3_music3_output_free(output);
        m3_music3_test_fixture_dispose(&fixture);
        return;
    }
    preserved = output;
    semantic_entry = fixture.semantic_rng_entry;
    semantic_exit = fixture.semantic_rng_exit;
    flow_entry = fixture.flow_rng_entry;
    flow_exit = fixture.flow_rng_exit;
    for (index = 0U; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        m3_music3_test_progress progress;
        m3_backend_allocation_stats after = {0U, 0U, 0U, 0U};
        bool expect_semantic =
            cases[index].phase > M3_MUSIC3_PHASE_SEMANTIC ||
            (cases[index].phase == M3_MUSIC3_PHASE_SEMANTIC &&
             cases[index].completed > 0U);
        bool expect_flow = cases[index].phase > M3_MUSIC3_PHASE_FLOW ||
            (cases[index].phase == M3_MUSIC3_PHASE_FLOW &&
             cases[index].completed > 0U);

        m3_music3_test_request(&request, cases[index].frames);
        request_copy = request;
        fixture.scripted_frames = cases[index].frames;
        fixture.semantic_rng_observed = false;
        fixture.flow_rng_observed = false;
        m3_music3_test_progress_init(&progress);
        progress.cancel_enabled = true;
        progress.cancel_phase = cases[index].phase;
        progress.cancel_completed = cases[index].completed;
        status = m3_music3_generate(
            fixture.engine, &request, m3_music3_test_progress_call,
            &progress, &output, &error);
        M3_TEST_EXPECT(
            test,
            status == M3_STATUS_CANCELLED && output == preserved &&
                progress.contract_valid &&
                ((expect_semantic && fixture.semantic_rng_observed &&
                  m3_music3_rng_equal(
                      &fixture.semantic_rng_entry, &semantic_entry) &&
                  m3_music3_rng_equal(
                      &fixture.semantic_rng_exit, &semantic_exit)) ||
                 (!expect_semantic && !fixture.semantic_rng_observed)) &&
                ((expect_flow && fixture.flow_rng_observed &&
                  m3_music3_rng_equal(
                      &fixture.flow_rng_entry, &flow_entry) &&
                  m3_music3_rng_equal(
                      &fixture.flow_rng_exit, &flow_exit)) ||
                (!expect_flow && !fixture.flow_rng_observed)) &&
                m3_music3_request_equal(&request, &request_copy) &&
                m3_music3_output_get_info(
                    output, &actual_info, &error) == M3_STATUS_OK &&
                m3_music3_output_info_equal(
                    &actual_info, &expected_info) &&
                m3_music3_output_read_channel_f32(
                    output, 0U, 0U, actual_zero, samples, &error) ==
                    M3_STATUS_OK &&
                m3_music3_output_read_channel_f32(
                    output, 1U, 0U, actual_one, samples, &error) ==
                    M3_STATUS_OK &&
                memcmp(actual_zero, expected_zero, bytes) == 0 &&
                memcmp(actual_one, expected_one, bytes) == 0 &&
                m3_backend_get_allocation_stats(
                    fixture.engine->backend, &after, &error) ==
                    M3_STATUS_OK &&
                m3_music3_stats_equal(&baseline, &after),
            "phase cancellation preserves request, output bits, and baseline");
    }
    m3_music3_test_request(&request, 1U);
    fixture.scripted_frames = 1U;
    fixture.semantic_rng_observed = false;
    fixture.flow_rng_observed = false;
    status = m3_music3_generate(
        fixture.engine, &request, NULL, NULL, &output, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OK && output != NULL &&
            fixture.semantic_rng_observed && fixture.flow_rng_observed &&
            m3_music3_rng_equal(
                &fixture.semantic_rng_entry, &semantic_entry) &&
            m3_music3_rng_equal(
                &fixture.semantic_rng_exit, &semantic_exit) &&
            m3_music3_rng_equal(&fixture.flow_rng_entry, &flow_entry) &&
            m3_music3_rng_equal(&fixture.flow_rng_exit, &flow_exit) &&
            m3_music3_output_get_info(
                output, &actual_info, &error) == M3_STATUS_OK &&
            m3_music3_output_info_equal(&actual_info, &expected_info) &&
            m3_music3_output_read_channel_f32(
                output, 0U, 0U, actual_zero, samples, &error) ==
                M3_STATUS_OK &&
            m3_music3_output_read_channel_f32(
                output, 1U, 0U, actual_one, samples, &error) ==
                M3_STATUS_OK &&
            memcmp(actual_zero, expected_zero, bytes) == 0 &&
            memcmp(actual_one, expected_one, bytes) == 0,
        "retry from the same seed and sequence is bit deterministic");
    free(actual_one);
    free(actual_zero);
    free(expected_one);
    free(expected_zero);
    m3_music3_output_free(output);
    m3_music3_test_fixture_dispose(&fixture);
}

void m3_test_music3_ownership_atomicity(m3_test_context *test)
{
    static const struct {
        m3_music3_test_corruption corruption;
        uint64_t frames;
        m3_music3_phase phase;
        uint64_t total;
        const char *error_message;
    } aliases[] = {
        {M3_MUSIC3_TEST_ALIAS_OLD_STAGE, 1U,
         M3_MUSIC3_PHASE_STAGE_SEMANTIC,
         M3_MUSIC3_STAGE_SEMANTIC_BYTES,
         "Music3 stage aliases the prior output"},
        {M3_MUSIC3_TEST_ALIAS_OLD_SEMANTIC, 1U,
         M3_MUSIC3_PHASE_SEMANTIC, 97U,
         "Music3 semantic output aliases an input"},
        {M3_MUSIC3_TEST_ALIAS_OLD_FLOW, 1U,
         M3_MUSIC3_PHASE_FLOW, 30U,
         "Music3 flow output aliases an input"},
        {M3_MUSIC3_TEST_ALIAS_OLD_RUNTIME, 1U,
         M3_MUSIC3_PHASE_MATERIALIZE_VOCODER,
         M3_VOCODER_SOURCE_WEIGHT_COUNT,
         "Music3 vocoder runtime aliases an input"},
        {M3_MUSIC3_TEST_ALIAS_OLD_DECODE, 1U,
         M3_MUSIC3_PHASE_DECODE,
         M3_VOCODER_DECODE_OPERATION_COUNT,
         "Music3 decoded output aliases prior storage"},
        {M3_MUSIC3_TEST_ALIAS_OLD_ASSEMBLE, 1U,
         M3_MUSIC3_PHASE_ASSEMBLE, 2U,
         "Music3 assembled output aliases a chunk"},
    };
    m3_music3_test_fixture fixture;
    m3_music3_test_progress progress;
    m3_music3_request request;
    m3_music3_output *output = NULL;
    m3_music3_output *preserved;
    m3_backend_allocation_stats baseline = {0U, 0U, 0U, 0U};
    m3_backend_allocation_stats after = {0U, 0U, 0U, 0U};
    float expected[4];
    float actual[4];
    m3_error error;
    m3_status status = m3_music3_test_fixture_create(
        &fixture, false, &error);
    size_t index;

    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create ownership fixture");
    if (status != M3_STATUS_OK) {
        return;
    }
    fixture.scripted_frames = 300U;
    m3_music3_test_request(&request, 300U);
    status = m3_music3_generate(
        fixture.engine, &request, NULL, NULL, &output, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OK && output != NULL &&
            m3_music3_output_read_channel_f32(
                output, 1U, 0U, expected, 4U, &error) == M3_STATUS_OK &&
            m3_backend_get_allocation_stats(
                fixture.engine->backend, &baseline, &error) ==
                M3_STATUS_OK,
        "seed old output before ownership matrix");
    if (status != M3_STATUS_OK || output == NULL) {
        m3_music3_output_free(output);
        m3_music3_test_fixture_dispose(&fixture);
        return;
    }
    preserved = output;
    fixture.alias_storage = output->waveform.storage;
    fixture.scripted_frames = 1U;
    m3_music3_test_request(&request, 1U);
    for (index = 0U; index < sizeof(aliases) / sizeof(aliases[0]); ++index) {
        char message[192];
        bool ownership_message;

        fixture.corruption = aliases[index].corruption;
        m3_music3_test_progress_init(&progress);
        m3_error_reset(&error);
        status = m3_music3_generate(
            fixture.engine, &request, m3_music3_test_progress_call,
            &progress, &output, &error);
        ownership_message =
            strcmp(error.message, aliases[index].error_message) == 0;
        (void)m3_backend_get_allocation_stats(
            fixture.engine->backend, &after, &error);
        (void)snprintf(
            message, sizeof(message),
            "old-output alias %d: status %d pointer %d contract %d final %d bits %d stats %d",
            (int)aliases[index].corruption, (int)status,
            output == preserved, progress.contract_valid,
            m3_music3_test_phase_reached(
                &progress, aliases[index].phase, aliases[index].total),
            output != NULL && m3_music3_output_read_channel_f32(
                                  output, 1U, 0U, actual, 4U, &error) ==
                                  M3_STATUS_OK &&
                memcmp(actual, expected, sizeof(actual)) == 0,
            m3_music3_stats_equal(&baseline, &after));
        M3_TEST_EXPECT(
            test, status == M3_STATUS_INTERNAL && output == preserved &&
                progress.contract_valid &&
                ownership_message &&
                !m3_music3_test_phase_reached(
                    &progress, aliases[index].phase,
                    aliases[index].total) &&
                m3_music3_output_read_channel_f32(
                    output, 1U, 0U, actual, 4U, &error) ==
                    M3_STATUS_OK &&
                memcmp(actual, expected, sizeof(actual)) == 0 &&
                m3_music3_stats_equal(&baseline, &after), message);
    }
    fixture.corruption = M3_MUSIC3_TEST_ALIAS_FIRST_STAGE;
    m3_music3_test_progress_init(&progress);
    status = m3_music3_generate(
        fixture.engine, &request, m3_music3_test_progress_call,
        &progress, &output, &error);
    (void)m3_backend_get_allocation_stats(
        fixture.engine->backend, &after, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_INTERNAL && output == preserved &&
            progress.contract_valid &&
            !m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_STAGE_SEMANTIC,
                M3_MUSIC3_STAGE_SEMANTIC_BYTES) &&
            m3_music3_output_read_channel_f32(
                output, 1U, 0U, actual, 4U, &error) == M3_STATUS_OK &&
            memcmp(actual, expected, sizeof(actual)) == 0 &&
            m3_music3_stats_equal(&baseline, &after),
            "second stage cannot claim the first stage owner");
    fixture.corruption = M3_MUSIC3_TEST_ALIAS_OLD_STAGE;
    m3_music3_test_progress_init(&progress);
    status = m3_music3_generate(
        fixture.engine, &request, m3_music3_test_progress_call,
        &progress, &output, NULL);
    (void)m3_backend_get_allocation_stats(
        fixture.engine->backend, &after, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_INTERNAL && output == preserved &&
            progress.contract_valid &&
            m3_music3_output_read_channel_f32(
                output, 1U, 0U, actual, 4U, &error) == M3_STATUS_OK &&
            memcmp(actual, expected, sizeof(actual)) == 0 &&
            m3_music3_stats_equal(&baseline, &after),
        "old output remains atomic when ownership error is null");
    m3_music3_output_free(output);
    m3_music3_test_fixture_dispose(&fixture);
}
