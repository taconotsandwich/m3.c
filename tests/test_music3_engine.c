/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "music3_engine_test.h"

#include "m3_waveform_internal.h"

#include <stdio.h>
#include <string.h>

static bool m3_music3_test_final_totals(
    const m3_music3_test_progress *progress, uint64_t frames,
    size_t chunks)
{
    const uint64_t totals[M3_MUSIC3_PHASE_COUNT] = {
        1U,
        M3_MUSIC3_STAGE_SEMANTIC_BYTES,
        49U * frames + 48U,
        M3_MUSIC3_STAGE_FLOW_BYTES,
        (uint64_t)chunks * M3_FLOW_INFERENCE_STEPS,
        M3_MUSIC3_STAGE_VOCODER_BYTES,
        M3_VOCODER_SOURCE_WEIGHT_COUNT,
        (uint64_t)chunks * M3_VOCODER_DECODE_OPERATION_COUNT,
        (uint64_t)chunks * 2U};
    m3_music3_phase phase;

    if (!progress->contract_valid) {
        return false;
    }
    for (phase = M3_MUSIC3_PHASE_PREPARE;
         phase < M3_MUSIC3_PHASE_COUNT;
         phase = (m3_music3_phase)((int)phase + 1)) {
        size_t first = SIZE_MAX;
        size_t last = SIZE_MAX;
        size_t index;

        for (index = 0U; index < progress->count; ++index) {
            if (progress->phase[index] == phase) {
                if (first == SIZE_MAX) {
                    first = index;
                }
                last = index;
                if (progress->total[index] != totals[(size_t)phase]) {
                    return false;
                }
            }
        }
        if (first == SIZE_MAX || progress->completed[first] != 0U ||
            progress->completed[last] != totals[(size_t)phase]) {
            return false;
        }
    }
    return true;
}

static bool m3_music3_test_all_stages_once(
    const m3_music3_test_fixture *fixture)
{
    size_t index;

    for (index = 0U; index < M3_MUSIC3_WEIGHT_COMPONENT_COUNT; ++index) {
        if (fixture->stage_calls[index] != 1U) {
            return false;
        }
    }
    return true;
}

static bool m3_music3_test_wav(
    const m3_music3_output *output, m3_error *error)
{
    static const m3_test_fixture empty = {"empty", NULL, 0U};
    m3_test_temp_file file;
    uint8_t riff[4];
    FILE *stream;
    bool success;

    if (!m3_test_temp_file_create(&file, &empty)) {
        return false;
    }
    success = m3_music3_output_write_wav(
                  output, file.path, error) == M3_STATUS_OK;
    stream = success ? fopen(file.path, "rb") : NULL;
    if (stream == NULL || fread(riff, 1U, sizeof(riff), stream) !=
                              sizeof(riff) ||
        memcmp(riff, "RIFF", sizeof(riff)) != 0) {
        success = false;
    }
    if (stream != NULL && fclose(stream) != 0) {
        success = false;
    }
    if (!m3_test_temp_file_remove(&file)) {
        success = false;
    }
    return success;
}

void m3_test_music3_scripted_pipeline(m3_test_context *test)
{
    m3_music3_test_fixture fixture;
    m3_music3_test_progress progress;
    m3_music3_request request;
    m3_music3_output_info info = {0U, 0U, 0U};
    m3_music3_output *output = NULL;
    m3_waveform_measurement measurement = {0U, 0U, 0U};
    m3_backend_allocation_stats before = {0U, 0U, 0U, 0U};
    m3_backend_allocation_stats after = {0U, 0U, 0U, 0U};
    float channel_zero[8];
    float channel_one[8];
    m3_error error;
    m3_status status = m3_music3_test_fixture_create(
        &fixture, false, &error);

    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create scripted Music3 fixture");
    if (status != M3_STATUS_OK) {
        return;
    }
    fixture.scripted_frames = 300U;
    m3_music3_test_request(&request, 300U);
    m3_music3_test_progress_init(&progress);
    (void)m3_backend_get_allocation_stats(
        fixture.engine->backend, &before, &error);
    status = m3_music3_generate(
        fixture.engine, &request, m3_music3_test_progress_call,
        &progress, &output, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_OK && output != NULL,
                   "run one serial Music3 scheduler");
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OK && fixture.prompt_rows_valid &&
            m3_music3_test_all_stages_once(&fixture) &&
            fixture.semantic_calls == 1U && fixture.flow_calls == 1U &&
            fixture.materialize_calls == 1U &&
            fixture.runtime_validate_calls == 1U &&
            fixture.decode_calls == 2U && fixture.assemble_calls == 1U &&
            m3_music3_test_final_totals(&progress, 300U, 2U) &&
            m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_STAGE_SEMANTIC,
                UINT64_C(17168951296)) &&
            m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_STAGE_FLOW,
                UINT64_C(100671524)) &&
            m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_DECODE, 73U) &&
            m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_ASSEMBLE, 2U),
        "scheduler uses every mandatory operation and exact phase total");
    M3_TEST_EXPECT(
        test,
        m3_waveform_measure(300U, 2U, &measurement, &error) ==
                M3_STATUS_OK &&
            m3_music3_output_get_info(output, &info, &error) ==
                M3_STATUS_OK &&
            info.sample_rate == M3_MUSIC3_SAMPLE_RATE &&
            info.channel_count == M3_MUSIC3_CHANNEL_COUNT &&
            info.samples_per_channel == measurement.output_samples,
        "public output info reports exact assembled waveform shape");
    (void)memset(channel_zero, 0, sizeof(channel_zero));
    (void)memset(channel_one, 0, sizeof(channel_one));
    M3_TEST_EXPECT(
        test,
        m3_music3_output_read_channel_f32(
            output, 0U, 0U, channel_zero,
            sizeof(channel_zero) / sizeof(channel_zero[0]), &error) ==
                M3_STATUS_OK,
        "read a bounded nonempty range from planar channel zero");
    M3_TEST_EXPECT(
        test,
        m3_music3_output_read_channel_f32(
            output, 1U, 0U, channel_one,
            sizeof(channel_one) / sizeof(channel_one[0]), &error) ==
            M3_STATUS_OK,
        "read a bounded nonempty range from planar channel one");
    M3_TEST_EXPECT(
        test,
        channel_zero[0] == 0.125F && channel_zero[1] == 0.1875F &&
            channel_one[0] == -0.25F && channel_one[1] == -0.3125F,
        "channel reads preserve distinct bit-exact planar values");
    M3_TEST_EXPECT(
        test,
        m3_music3_output_read_channel_f32(
            output, 1U, info.samples_per_channel, NULL, 0U,
            &error) == M3_STATUS_OK &&
            m3_music3_output_read_channel_f32(
                output, 2U, 0U, channel_zero, 1U, &error) ==
                M3_STATUS_OUT_OF_RANGE,
        "bounded channel reads validate zero and bad ranges");
    M3_TEST_EXPECT(test, m3_music3_test_wav(output, &error),
                   "public WAVE path writes through the tensor writer");
    fixture.scripted_frames = 1U;
    m3_music3_test_request(&request, 300U);
    m3_music3_test_progress_init(&progress);
    status = m3_music3_generate(
        fixture.engine, &request, m3_music3_test_progress_call,
        &progress, &output, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OK && output != NULL &&
            m3_music3_test_final_totals(&progress, 300U, 1U) &&
            m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_SEMANTIC, 14748U) &&
            m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_FLOW, 30U) &&
            m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_DECODE, 73U) &&
            m3_music3_test_phase_reached(
                &progress, M3_MUSIC3_PHASE_ASSEMBLE, 2U) &&
            m3_waveform_measure(1U, 1U, &measurement, &error) ==
                M3_STATUS_OK &&
            m3_music3_output_get_info(output, &info, &error) ==
                M3_STATUS_OK &&
            info.samples_per_channel == measurement.output_samples,
        "maximum frames governs semantic progress while actual frames govern downstream output");
    m3_music3_output_free(output);
    output = NULL;
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(
            fixture.engine->backend, &after, &error) == M3_STATUS_OK &&
            after.live_allocated_bytes == before.live_allocated_bytes &&
            after.live_storage_count == before.live_storage_count,
        "successful scheduler cleanup returns to backend baseline");
    m3_music3_test_fixture_dispose(&fixture);
}

void m3_test_music3_reduced_metal_pipeline(m3_test_context *test)
{
    m3_music3_test_fixture fixture;
    m3_music3_test_progress progress;
    m3_music3_request request;
    m3_music3_output *output = NULL;
    m3_backend_allocation_stats before = {0U, 0U, 0U, 0U};
    m3_backend_allocation_stats after = {0U, 0U, 0U, 0U};
    m3_backend_info backend_info = {{0}, M3_BACKEND_HOST, false, 0U, 0U};
    m3_error error;
    m3_status status = m3_music3_test_fixture_create(
        &fixture, true, &error);

    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create real-Metal Music3 fixture");
    if (status != M3_STATUS_OK) {
        return;
    }
    m3_music3_test_request(&request, 1U);
    m3_music3_test_progress_init(&progress);
    (void)m3_backend_get_allocation_stats(
        fixture.engine->backend, &before, &error);
    status = m3_music3_generate(
        fixture.engine, &request, m3_music3_test_progress_call,
        &progress, &output, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_OK && output != NULL,
                   "run high-level scheduler on one Metal backend");
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OK && fixture.prompt_rows_valid &&
            m3_backend_get_info(
                fixture.engine->backend, &backend_info, &error) ==
                M3_STATUS_OK &&
            backend_info.kind == M3_BACKEND_METAL &&
            m3_music3_test_all_stages_once(&fixture) &&
            fixture.semantic.prefill_calls == 1U &&
            fixture.semantic.decode_calls == 2U &&
            fixture.semantic.feedback_calls == 1U &&
            fixture.semantic.advance_calls == 1U &&
            fixture.semantic.contract_valid && fixture.flow_calls == 1U &&
            fixture.materialize_calls == 1U &&
            fixture.runtime_validate_calls == 1U &&
            fixture.decode_calls == 1U && fixture.assemble_calls == 1U &&
            m3_music3_test_final_totals(&progress, 1U, 1U),
        "Metal fixture traverses semantic scheduler, flow, 91 weights, "
        "73 decode ops, and assembly");
    m3_music3_output_free(output);
    output = NULL;
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(
            fixture.engine->backend, &after, &error) == M3_STATUS_OK &&
            after.live_allocated_bytes == before.live_allocated_bytes &&
            after.live_storage_count == before.live_storage_count,
        "real-Metal scheduler returns to fixture allocation baseline");
    m3_music3_test_fixture_dispose(&fixture);
}

typedef struct {
    m3_music3_engine *engine;
    const m3_music3_request *request;
    bool attempted;
    m3_status nested_status;
} m3_music3_reentrant_context;

static bool m3_music3_reentrant_progress(
    void *context, m3_music3_phase phase, uint64_t completed,
    uint64_t total)
{
    m3_music3_reentrant_context *nested = context;

    (void)total;
    if (!nested->attempted && phase == M3_MUSIC3_PHASE_PREPARE &&
        completed == 0U) {
        m3_music3_output *output = NULL;

        nested->attempted = true;
        nested->nested_status = m3_music3_generate(
            nested->engine, nested->request, NULL, NULL, &output, NULL);
    }
    return true;
}

void m3_test_music3_reentrant_and_cross_engine(m3_test_context *test)
{
    m3_music3_test_fixture first;
    m3_music3_test_fixture second;
    m3_music3_request request;
    m3_music3_output *output = NULL;
    m3_music3_output *preserved;
    m3_music3_reentrant_context nested;
    m3_error error;
    m3_status first_status = m3_music3_test_fixture_create(
        &first, false, &error);
    m3_status second_status = m3_music3_test_fixture_create(
        &second, false, &error);

    M3_TEST_EXPECT(test,
                   first_status == M3_STATUS_OK &&
                       second_status == M3_STATUS_OK,
                   "create two independent Music3 engines");
    if (first_status != M3_STATUS_OK || second_status != M3_STATUS_OK) {
        m3_music3_test_fixture_dispose(&first);
        m3_music3_test_fixture_dispose(&second);
        return;
    }
    m3_music3_test_request(&request, 1U);
    nested = (m3_music3_reentrant_context){
        first.engine, &request, false, M3_STATUS_OK};
    M3_TEST_EXPECT(
        test,
        m3_music3_generate(
            first.engine, &request, m3_music3_reentrant_progress, &nested,
            &output, &error) == M3_STATUS_OK &&
            nested.attempted &&
            nested.nested_status == M3_STATUS_INVALID_ARGUMENT,
        "nested generation rejects while the outer scheduler completes");
    preserved = output;
    M3_TEST_EXPECT(
        test,
        m3_music3_generate(
            second.engine, &request, NULL, NULL, &output, &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            output == preserved,
        "cross-engine replacement rejects without touching old output");
    m3_music3_output_free(output);
    m3_music3_test_fixture_dispose(&second);
    m3_music3_test_fixture_dispose(&first);
}
