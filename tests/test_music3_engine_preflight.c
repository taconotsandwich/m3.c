/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend_internal.h"
#include "m3_flow_internal.h"
#include "m3_music3_schema.h"
#include "music3_engine_test.h"

#include <stdlib.h>
#include <string.h>

#define M3_PREFLIGHT_LIVE_SENTINEL_BYTES UINT64_C(64)
#define M3_PREFLIGHT_TEST_LIMIT UINT64_C(30000000000)
#define M3_PREFLIGHT_BACKING_CAP (1024U * 1024U)

typedef struct {
    size_t backing_cap;
} m3_preflight_backend_context;

static void m3_preflight_backend_destroy(void *pointer)
{
    free(pointer);
}

static m3_status m3_preflight_backend_allocate(
    void *pointer, size_t byte_count, size_t alignment, void **handle,
    void **data, m3_error *error)
{
    const m3_preflight_backend_context *context = pointer;
    size_t allocation_size = byte_count;
    size_t remainder;
    void *memory;

    *handle = NULL;
    *data = NULL;
    if (byte_count == 0U) {
        return M3_STATUS_OK;
    }
    if (allocation_size > context->backing_cap) {
        allocation_size = context->backing_cap;
    }
    remainder = allocation_size & (alignment - 1U);
    if (remainder != 0U) {
        if (allocation_size > SIZE_MAX - (alignment - remainder)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "sparse preflight allocation overflows");
        }
        allocation_size += alignment - remainder;
    }
    memory = aligned_alloc(alignment, allocation_size);
    if (memory == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "sparse preflight allocation failed");
    }
    *handle = memory;
    *data = memory;
    return M3_STATUS_OK;
}

static void m3_preflight_backend_free(
    void *pointer, void *handle, void *data)
{
    (void)pointer;
    (void)data;
    free(handle);
}

static m3_status m3_preflight_backend_execute(
    void *pointer, const m3_command *commands, size_t command_count,
    m3_scratch_arena *scratch, m3_error *error)
{
    (void)pointer;
    (void)commands;
    (void)command_count;
    (void)scratch;
    return m3_error_set(error, M3_STATUS_UNSUPPORTED,
                        "sparse preflight backend does not execute");
}

static m3_status m3_preflight_backend_create(
    uint64_t recommended, m3_backend **backend, m3_error *error)
{
    static const m3_backend_vtable vtable = {
        m3_preflight_backend_destroy,
        m3_preflight_backend_allocate,
        m3_preflight_backend_free,
        m3_preflight_backend_execute};
    m3_preflight_backend_context *context = calloc(1U, sizeof(*context));
    m3_backend_info info;
    m3_status status;

    if (context == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate sparse preflight backend");
    }
    context->backing_cap = M3_PREFLIGHT_BACKING_CAP;
    (void)memset(&info, 0, sizeof(info));
    (void)memcpy(info.name, "Music3 sparse preflight",
                 sizeof("Music3 sparse preflight"));
    info.kind = M3_BACKEND_HOST;
    info.unified_memory = true;
    info.maximum_storage_bytes = UINT64_MAX;
    info.recommended_working_set_bytes = recommended;
    status = m3_backend_create_internal(
        &vtable, context, &info, backend, error);
    if (status != M3_STATUS_OK) {
        free(context);
    }
    return status;
}

static bool m3_preflight_engine_init(
    m3_music3_engine *engine, uint64_t recommended, m3_error *error)
{
    size_t index;

    (void)memset(engine, 0, sizeof(*engine));
    if (m3_preflight_backend_create(
            recommended, &engine->backend, error) != M3_STATUS_OK) {
        return false;
    }
    for (index = 0U; index < M3_MUSIC3_WEIGHT_COMPONENT_COUNT; ++index) {
        m3_music3_schema_summary expected;

        if (m3_music3_schema_expected_summary(
                (m3_component_id)index, &expected, error) !=
                M3_STATUS_OK) {
            m3_backend_free(engine->backend);
            engine->backend = NULL;
            return false;
        }
        engine->component_payload_bytes[index] = expected.payload_bytes;
        engine->component_largest_shard_bytes[index] =
            expected.payload_bytes;
    }
    return true;
}

static m3_status m3_preflight_storage(
    m3_backend *backend, uint64_t bytes, m3_storage **storage,
    m3_error *error)
{
    if (bytes > SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "preflight test storage exceeds size_t");
    }
    return m3_storage_allocate(
        backend, (size_t)bytes, 64U, storage, error);
}

static bool m3_preflight_semantic_boundary(
    uint64_t frame_limit, uint64_t exact_without_sentinel,
    bool one_byte_short)
{
    m3_music3_engine engine;
    m3_semantic_plan plan;
    m3_storage *sentinel = NULL;
    m3_storage *prompt = NULL;
    uint64_t recommended = exact_without_sentinel +
        M3_PREFLIGHT_LIVE_SENTINEL_BYTES - (one_byte_short ? 1U : 0U);
    uint64_t planned;
    m3_error error;
    m3_status status;
    bool valid = m3_preflight_engine_init(
        &engine, recommended, &error);

    if (valid) {
        valid = m3_preflight_storage(
                    engine.backend, M3_PREFLIGHT_LIVE_SENTINEL_BYTES,
                    &sentinel, &error) == M3_STATUS_OK &&
                m3_preflight_storage(
                    engine.backend, 8U * 5000U, &prompt, &error) ==
                    M3_STATUS_OK &&
                m3_semantic_plan_build(
                    5000U, frame_limit, &plan, &error) == M3_STATUS_OK;
    }
    if (valid) {
        planned = plan.maximum_added_bytes +
            engine.component_payload_bytes[M3_COMPONENT_LANGUAGE_MODEL] +
            engine.component_payload_bytes[
                M3_COMPONENT_RVQ_DEPTH_DECODER] +
            8U * 5000U;
        status = m3_music3_preflight_semantic(&engine, &plan, &error);
        valid = planned == exact_without_sentinel &&
            status == (one_byte_short ? M3_STATUS_OUT_OF_MEMORY :
                                        M3_STATUS_OK);
    }
    m3_storage_free(prompt);
    m3_storage_free(sentinel);
    m3_backend_free(engine.backend);
    return valid;
}

static bool m3_preflight_flow_boundary(bool one_byte_short)
{
    const uint64_t shape[] = {1U, 1U, M3_RVQ_CODEBOOK_COUNT,
                              M3_QWEN_HIDDEN_SIZE};
    const uint64_t exact = UINT64_C(9831738948);
    m3_music3_engine engine;
    m3_storage *sentinel = NULL;
    m3_storage *semantic = NULL;
    m3_tensor_view frames;
    m3_error error;
    m3_status status = M3_STATUS_INTERNAL;
    bool valid = m3_preflight_engine_init(
        &engine,
        exact + M3_PREFLIGHT_LIVE_SENTINEL_BYTES -
            (one_byte_short ? 1U : 0U),
        &error);

    m3_tensor_view_init(&frames);
    if (valid) {
        valid = m3_preflight_storage(
                    engine.backend, M3_PREFLIGHT_LIVE_SENTINEL_BYTES,
                    &sentinel, &error) == M3_STATUS_OK &&
                m3_preflight_storage(
                    engine.backend,
                    2U * M3_RVQ_CODEBOOK_COUNT * M3_QWEN_HIDDEN_SIZE,
                    &semantic, &error) == M3_STATUS_OK &&
                m3_tensor_view_contiguous(
                    &frames, semantic, M3_DTYPE_BF16, 4U, shape, 0U,
                    &error) == M3_STATUS_OK;
    }
    if (valid) {
        status = m3_music3_preflight_flow(&engine, &frames, &error);
        valid = status == (one_byte_short ? M3_STATUS_OUT_OF_MEMORY :
                                            M3_STATUS_OK);
    }
    m3_storage_free(semantic);
    m3_storage_free(sentinel);
    m3_backend_free(engine.backend);
    return valid;
}

static bool m3_preflight_vocoder_boundary(bool one_byte_short)
{
    const uint64_t exact = UINT64_C(433315084);
    m3_music3_engine engine;
    m3_flow_output latents;
    m3_flow_config config;
    m3_storage *sentinel = NULL;
    uint64_t start = 0U;
    uint64_t frames = 0U;
    uint64_t length = 0U;
    uint64_t shape[3];
    m3_error error;
    m3_status status = M3_STATUS_INTERNAL;
    bool valid = m3_preflight_engine_init(
        &engine,
        exact + M3_PREFLIGHT_LIVE_SENTINEL_BYTES -
            (one_byte_short ? 1U : 0U),
        &error);

    m3_flow_output_init(&latents);
    m3_flow_config_init(&config);
    if (valid) {
        valid = m3_flow_chunk_window(
                    &config, 1U, 0U, &start, &frames, &error) ==
                    M3_STATUS_OK &&
                m3_condition_output_length(
                    frames, config.condition.resize_numerator,
                    config.condition.resize_denominator, &length,
                    &error) == M3_STATUS_OK;
    }
    if (valid) {
        latents.chunk_count = 1U;
        latents.storages = calloc(1U, sizeof(*latents.storages));
        latents.chunks = calloc(1U, sizeof(*latents.chunks));
        valid = latents.storages != NULL && latents.chunks != NULL;
    }
    shape[0] = 1U;
    shape[1] = M3_FLOW_LATENT_CHANNELS;
    shape[2] = length;
    if (valid) {
        valid = m3_preflight_storage(
                    engine.backend, M3_PREFLIGHT_LIVE_SENTINEL_BYTES,
                    &sentinel, &error) == M3_STATUS_OK &&
                m3_preflight_storage(
                    engine.backend,
                    M3_FLOW_LATENT_CHANNELS * length * sizeof(float),
                    &latents.storages[0], &error) == M3_STATUS_OK &&
                m3_tensor_view_contiguous(
                    &latents.chunks[0], latents.storages[0],
                    M3_DTYPE_F32, 3U, shape, 0U, &error) == M3_STATUS_OK;
    }
    if (valid) {
        status = m3_music3_preflight_vocoder(
            &engine, &latents, 1U, &error);
        valid = status == (one_byte_short ? M3_STATUS_OUT_OF_MEMORY :
                                            M3_STATUS_OK);
    }
    (void)start;
    m3_flow_output_dispose(&latents);
    m3_storage_free(sentinel);
    m3_backend_free(engine.backend);
    return valid;
}

typedef struct {
    m3_music3_test_progress progress;
    m3_music3_test_fixture *fixture;
    m3_music3_phase phase;
    uint64_t sentinel_bytes;
    m3_storage *sentinel;
    bool injected;
} m3_preflight_progress;

static bool m3_preflight_progress_call(
    void *pointer, m3_music3_phase phase, uint64_t completed,
    uint64_t total)
{
    m3_preflight_progress *progress = pointer;
    m3_error error;

    if (!m3_music3_test_progress_call(
            &progress->progress, phase, completed, total)) {
        return false;
    }
    if (!progress->injected && phase == progress->phase &&
        completed == 0U) {
        progress->injected = true;
        if (m3_preflight_storage(
                progress->fixture->engine->backend,
                progress->sentinel_bytes, &progress->sentinel,
                &error) != M3_STATUS_OK) {
            return false;
        }
    }
    return true;
}

static bool m3_preflight_prompt_total(
    m3_music3_test_fixture *fixture, const m3_music3_request *request,
    uint64_t *total, m3_error *error)
{
    m3_music3_prepared_prompt prompt;
    m3_status status;

    m3_music3_prepared_prompt_init(&prompt);
    status = m3_music3_prepare_prompt(
        fixture->engine, request, &prompt, error);
    if (status == M3_STATUS_OK) {
        *total = (uint64_t)m3_storage_size(prompt.storage) +
            prompt.semantic_plan.maximum_added_bytes +
            fixture->engine->component_payload_bytes[
                M3_COMPONENT_LANGUAGE_MODEL] +
            fixture->engine->component_payload_bytes[
                M3_COMPONENT_RVQ_DEPTH_DECODER];
    }
    m3_music3_prepared_prompt_dispose(&prompt);
    return status == M3_STATUS_OK;
}

static bool m3_preflight_high_level_case(
    m3_music3_phase phase, uint64_t planned_without_sentinel,
    m3_component_id first_component, bool one_byte_short)
{
    m3_music3_test_fixture fixture;
    m3_music3_request request;
    m3_music3_output *output = NULL;
    m3_preflight_progress progress = {0};
    m3_backend *backend = NULL;
    m3_backend_allocation_stats after;
    m3_error error;
    m3_status status = m3_music3_test_fixture_create(
        &fixture, false, &error);
    bool valid = status == M3_STATUS_OK &&
        planned_without_sentinel < M3_PREFLIGHT_TEST_LIMIT;

    if (valid) {
        status = m3_preflight_backend_create(
            M3_PREFLIGHT_TEST_LIMIT, &backend, &error);
        valid = status == M3_STATUS_OK;
    }
    if (valid) {
        m3_backend_free(fixture.engine->backend);
        fixture.engine->backend = backend;
        backend = NULL;
        m3_music3_test_request(&request, 1U);
        (void)memset(&progress, 0, sizeof(progress));
        m3_music3_test_progress_init(&progress.progress);
        progress.fixture = &fixture;
        progress.phase = phase;
        progress.sentinel_bytes =
            M3_PREFLIGHT_TEST_LIMIT - planned_without_sentinel +
            (one_byte_short ? 1U : 0U);
        status = m3_music3_generate(
            fixture.engine, &request, m3_preflight_progress_call,
            &progress, &output, &error);
        valid = progress.injected && progress.progress.contract_valid &&
            (one_byte_short
                 ? status == M3_STATUS_OUT_OF_MEMORY && output == NULL &&
                       fixture.stage_calls[first_component] == 0U &&
                       !m3_music3_test_phase_reached(
                           &progress.progress, phase,
                           progress.progress.total[
                               progress.progress.count - 1U])
                 : status == M3_STATUS_OK && output != NULL &&
                       fixture.stage_calls[first_component] == 1U);
    }
    m3_music3_output_free(output);
    m3_storage_free(progress.sentinel);
    if (valid) {
        valid = m3_backend_get_allocation_stats(
                    fixture.engine->backend, &after, &error) ==
                    M3_STATUS_OK &&
                after.live_allocated_bytes == 0U &&
                after.live_storage_count == 0U;
    }
    m3_backend_free(backend);
    m3_music3_test_fixture_dispose(&fixture);
    return valid;
}

void m3_test_music3_aggregate_preflights(m3_test_context *test)
{
    m3_music3_test_fixture probe;
    m3_music3_request request;
    uint64_t semantic_total = 0U;
    m3_error error;
    m3_status status = m3_music3_test_fixture_create(
        &probe, false, &error);

    M3_TEST_EXPECT(
        test,
        m3_preflight_semantic_boundary(
            9000U, UINT64_C(23801926732), false) &&
            m3_preflight_semantic_boundary(
                9000U, UINT64_C(23801926732), true) &&
            m3_preflight_semantic_boundary(
                1U, UINT64_C(21145709900), false) &&
            m3_preflight_semantic_boundary(
                1U, UINT64_C(21145709900), true),
        "semantic aggregate includes retained 8T prompt and unrelated live bytes");
    M3_TEST_EXPECT(
        test,
        m3_preflight_flow_boundary(false) &&
            m3_preflight_flow_boundary(true),
        "flow aggregate passes exactly and fails one byte short");
    M3_TEST_EXPECT(
        test,
        m3_preflight_vocoder_boundary(false) &&
            m3_preflight_vocoder_boundary(true),
        "vocoder aggregate passes exactly and fails one byte short");
    if (status == M3_STATUS_OK) {
        m3_music3_test_request(&request, 1U);
        if (!m3_preflight_prompt_total(
                &probe, &request, &semantic_total, &error)) {
            status = error.status;
        }
    }
    m3_music3_test_fixture_dispose(&probe);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OK &&
            m3_preflight_high_level_case(
                M3_MUSIC3_PHASE_PREPARE, semantic_total,
                M3_COMPONENT_LANGUAGE_MODEL, false) &&
            m3_preflight_high_level_case(
                M3_MUSIC3_PHASE_PREPARE, semantic_total,
                M3_COMPONENT_LANGUAGE_MODEL, true) &&
            m3_preflight_high_level_case(
                M3_MUSIC3_PHASE_STAGE_FLOW, UINT64_C(9831738948),
                M3_COMPONENT_CONDITION_ENCODER, false) &&
            m3_preflight_high_level_case(
                M3_MUSIC3_PHASE_STAGE_FLOW, UINT64_C(9831738948),
                M3_COMPONENT_CONDITION_ENCODER, true) &&
            m3_preflight_high_level_case(
                M3_MUSIC3_PHASE_STAGE_VOCODER, UINT64_C(433315084),
                M3_COMPONENT_VOCODER, false) &&
            m3_preflight_high_level_case(
                M3_MUSIC3_PHASE_STAGE_VOCODER, UINT64_C(433315084),
                M3_COMPONENT_VOCODER, true),
        "scheduler preflights before each stage at exact and one-byte-short limits");
}
