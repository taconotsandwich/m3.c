/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "semantic_runtime_test.h"

#include "m3_backend.h"

#include <stdint.h>
#include <string.h>

static bool m3_semantic_schedule_progress_exact(
    const m3_semantic_test_progress *progress, uint64_t total)
{
    size_t index;

    if (!progress->total_stable || progress->total != total ||
        progress->count != (size_t)(total + 1U)) {
        return false;
    }
    for (index = 0U; index < progress->count; ++index) {
        if (progress->values[index] != (uint64_t)index) {
            return false;
        }
    }
    return true;
}

static bool m3_semantic_schedule_rng_exact(
    const m3_rng *initial, const m3_rng *actual,
    const m3_semantic_test_fixture *fixture, size_t attempts)
{
    m3_rng expected = *initial;
    m3_error error;
    float value;
    size_t attempt;
    size_t residual = 0U;

    m3_error_reset(&error);
    for (attempt = 0U; attempt < attempts; ++attempt) {
        size_t index;

        if (m3_rng_uniform_f32(&expected, &value, &error) !=
            M3_STATUS_OK) {
            return false;
        }
        for (index = 0U; index < M3_RVQ_RESIDUAL_COUNT; ++index) {
            if (m3_rng_uniform_f32(&expected, &value, &error) !=
                    M3_STATUS_OK ||
                memcmp(&value, &fixture->residual_uniforms[residual],
                       sizeof(value)) != 0) {
                return false;
            }
            residual += 1U;
        }
    }
    return residual == fixture->residual_uniform_count &&
           memcmp(&expected, actual, sizeof(expected)) == 0;
}

static bool m3_semantic_schedule_bit(
    const m3_semantic_output *output, uint64_t frame, uint64_t slot,
    uint16_t expected)
{
    uint64_t element =
        (frame * M3_RVQ_CODEBOOK_COUNT + slot) * M3_QWEN_HIDDEN_SIZE;
    uint16_t actual = 0U;
    m3_error error;

    m3_error_reset(&error);
    return output->storage != NULL &&
           m3_storage_read(
               output->storage, (size_t)element * sizeof(actual),
               &actual, sizeof(actual), &error) == M3_STATUS_OK &&
           actual == expected;
}

static bool m3_semantic_schedule_codes_exact(
    const m3_semantic_test_fixture *fixture, size_t attempts)
{
    size_t index;

    if (fixture->embedding_calls != attempts) {
        return false;
    }
    for (index = 0U; index < attempts; ++index) {
        if (fixture->embedded_codes[index] != (uint32_t)(index + 3U)) {
            return false;
        }
    }
    return true;
}

static void m3_semantic_schedule_case(m3_test_context *test,
                                      uint64_t frame_limit)
{
    m3_semantic_test_progress progress = {
        .cancel_at = UINT64_MAX
    };
    m3_semantic_test_fixture fixture;
    m3_semantic_output output;
    m3_backend *backend = NULL;
    m3_rng initial;
    m3_rng rng;
    m3_error error;
    uint64_t attempts = frame_limit + 1U;
    uint64_t total = 49U * frame_limit + 48U;
    bool ready;
    m3_status status;

    m3_error_reset(&error);
    ready = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
            m3_semantic_test_fixture_init(&fixture, backend) &&
            m3_rng_seed(&rng, UINT64_C(0x12345678),
                        UINT64_C(0x33445566), &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "create semantic Host schedule fixture");
    if (!ready) {
        m3_backend_free(backend);
        return;
    }
    m3_semantic_test_outcomes(
        &fixture, (size_t)attempts, SIZE_MAX);
    initial = rng;
    m3_semantic_output_init(&output);
    status = m3_semantic_generate_core(
        &fixture.operations, &fixture.prompt, frame_limit, &rng,
        m3_semantic_test_progress_callback, &progress, &output, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "complete semantic Host schedule");
    M3_TEST_EXPECT(
        test,
        fixture.start_calls == 1U && fixture.finish_calls == 1U &&
            fixture.prefill_calls == 1U &&
            fixture.embedding_calls == attempts &&
            fixture.decode_calls == attempts &&
            fixture.feedback_calls == frame_limit &&
            fixture.advance_calls == frame_limit &&
            fixture.cache_capacity == 5U + frame_limit &&
            fixture.contract_valid,
        "warmup, attempts, feedback, advance, and T+F cache are exact");
    M3_TEST_EXPECT(
        test,
        output.frame_hiddens.metadata.dtype == M3_DTYPE_BF16 &&
            output.frame_hiddens.metadata.rank == 4U &&
            output.frame_hiddens.metadata.shape[0] == 1U &&
            output.frame_hiddens.metadata.shape[1] == frame_limit &&
            output.frame_hiddens.metadata.shape[2] ==
                M3_RVQ_CODEBOOK_COUNT &&
            output.frame_hiddens.metadata.shape[3] ==
                M3_QWEN_HIDDEN_SIZE,
        "publish contiguous owned BF16 [1,F,8,4096]");
    M3_TEST_EXPECT(
        test,
        m3_semantic_schedule_bit(
            &output, 0U, 0U, UINT16_C(0x3f81)) &&
            m3_semantic_schedule_bit(
                &output, 0U, 1U, UINT16_C(0x4211)) &&
            m3_semantic_schedule_bit(
                &output, 0U, 7U, UINT16_C(0x4217)),
        "discard warmup and publish semantic slot before seven RVQ slots");
    if (frame_limit == 2U) {
        M3_TEST_EXPECT(
            test,
            m3_semantic_schedule_bit(
                &output, 1U, 0U, UINT16_C(0x3f82)) &&
                m3_semantic_schedule_bit(
                    &output, 1U, 1U, UINT16_C(0x4221)) &&
                m3_semantic_schedule_bit(
                    &output, 1U, 7U, UINT16_C(0x4227)),
            "preserve frame-major conditioning order at F=2");
    }
    M3_TEST_EXPECT(
        test,
        m3_semantic_schedule_rng_exact(
            &initial, &rng, &fixture, (size_t)attempts) &&
            m3_semantic_schedule_codes_exact(
                &fixture, (size_t)attempts),
        "draw semantic then seven residual uniforms on every attempt");
    M3_TEST_EXPECT(
        test, m3_semantic_schedule_progress_exact(&progress, total),
        "report one stable monotonic progress domain through the cap");
    m3_semantic_output_dispose(&output);
    m3_semantic_test_fixture_dispose(&fixture);
    m3_backend_free(backend);
}

void m3_test_semantic_schedule_caps(m3_test_context *test)
{
    m3_semantic_schedule_case(test, 1U);
    m3_semantic_schedule_case(test, 2U);
}

static void m3_semantic_guidance_case(m3_test_context *test,
                                      bool swapped,
                                      uint32_t expected_code)
{
    m3_semantic_test_fixture fixture;
    m3_semantic_output output;
    m3_backend_allocation_stats stats;
    m3_backend *backend = NULL;
    m3_rng rng;
    m3_error error;
    bool fixture_ready = false;
    bool ready;
    m3_status status;

    m3_error_reset(&error);
    m3_semantic_output_init(&output);
    ready = m3_backend_create_host(&backend, &error) == M3_STATUS_OK;
    if (ready) {
        fixture_ready = m3_semantic_test_fixture_init(&fixture, backend);
        ready = fixture_ready;
    }
    if (ready) {
        ready = m3_rng_seed(
                    &rng, UINT64_C(0xabcdef01),
                    UINT64_C(0x13579bdf), &error) == M3_STATUS_OK;
    }
    M3_TEST_EXPECT(test, ready,
                   "create adversarial semantic CFG fixture");
    if (!ready) {
        if (fixture_ready) {
            m3_semantic_test_fixture_dispose(&fixture);
        }
        m3_backend_free(backend);
        return;
    }
    m3_semantic_test_outcomes(&fixture, 2U, SIZE_MAX);
    m3_semantic_test_guidance_rows(&fixture, 7U, 19U, swapped);
    status = m3_semantic_generate_core(
        &fixture.operations, &fixture.prompt, 1U, &rng, NULL, NULL,
        &output, &error);
    M3_TEST_EXPECT(
        test,
        status == M3_STATUS_OK && fixture.embedding_calls == 2U &&
            fixture.embedded_codes[0] == expected_code &&
            fixture.embedded_codes[1] == expected_code,
        "production CFG sampler respects conditional and unconditional rows");
    m3_semantic_output_dispose(&output);
    m3_semantic_test_fixture_dispose(&fixture);
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 0U &&
            stats.live_storage_count == 0U,
        "adversarial CFG schedule releases every owned storage");
    m3_backend_free(backend);
}

void m3_test_semantic_guidance_row_order(m3_test_context *test)
{
    m3_semantic_guidance_case(test, false, 7U);
    m3_semantic_guidance_case(test, true, 19U);
}
