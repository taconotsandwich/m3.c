/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "semantic_runtime_test.h"

#include "m3_backend.h"

#include <stdint.h>
#include <string.h>

static bool m3_semantic_eos_progress_valid(
    const m3_semantic_test_progress *progress, uint64_t total,
    bool completed)
{
    size_t index;

    if (!progress->total_stable || progress->total != total ||
        progress->count == 0U || progress->values[0] != 0U) {
        return false;
    }
    for (index = 1U; index < progress->count; ++index) {
        if (progress->values[index] <= progress->values[index - 1U]) {
            return false;
        }
    }
    return !completed ||
           progress->values[progress->count - 1U] == total;
}

static bool m3_semantic_eos_rng_valid(
    const m3_rng *initial, const m3_rng *actual, size_t eos_index,
    bool success)
{
    m3_rng expected;

    if (!success) {
        return memcmp(initial, actual, sizeof(*initial)) == 0;
    }
    return m3_semantic_test_expected_rng(
               initial, eos_index * 8U + 1U, &expected) &&
           memcmp(&expected, actual, sizeof(expected)) == 0;
}

static void m3_semantic_eos_case(
    m3_test_context *test, uint64_t frame_limit, size_t eos_index,
    m3_status expected_status, uint64_t expected_frames)
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
    uint64_t total = frame_limit * 49U + 48U;
    bool success = expected_status == M3_STATUS_OK;
    bool ready;
    m3_status status;

    m3_error_reset(&error);
    ready = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
            m3_semantic_test_fixture_init(&fixture, backend) &&
            m3_rng_seed(&rng, UINT64_C(0x77665544),
                        UINT64_C(0x11223344), &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "create semantic EOS Host fixture");
    if (!ready) {
        m3_backend_free(backend);
        return;
    }
    m3_semantic_test_outcomes(&fixture, eos_index + 1U, eos_index);
    initial = rng;
    m3_semantic_output_init(&output);
    status = m3_semantic_generate_core(
        &fixture.operations, &fixture.prompt, frame_limit, &rng,
        m3_semantic_test_progress_callback, &progress, &output, &error);
    M3_TEST_EXPECT(test, status == expected_status,
                   "EOS schedule returns the expected status");
    M3_TEST_EXPECT(
        test,
        fixture.start_calls == 1U && fixture.finish_calls == 1U &&
            fixture.prefill_calls == 1U &&
            fixture.embedding_calls == eos_index &&
            fixture.decode_calls == eos_index &&
            fixture.feedback_calls == eos_index &&
            fixture.advance_calls == eos_index &&
            fixture.cache_capacity == 5U + frame_limit &&
            fixture.contract_valid,
        "EOS stops before embedding and preserves prior feedback advances");
    M3_TEST_EXPECT(
        test,
        m3_semantic_eos_rng_valid(
            &initial, &rng, eos_index, success),
        "EOS consumes one semantic draw only and commits RNG on success");
    M3_TEST_EXPECT(
        test,
        m3_semantic_eos_progress_valid(&progress, total, success),
        "EOS progress remains stable, monotonic, and completes on success");
    if (success) {
        M3_TEST_EXPECT(
            test,
            output.storage != NULL &&
                output.frame_hiddens.metadata.shape[1] ==
                    expected_frames,
            "EOS publishes only frames before its attempt");
    } else {
        M3_TEST_EXPECT(
            test,
            output.storage == NULL &&
                output.frame_hiddens.storage == NULL,
            "EOS before the first emitted frame leaves output atomic");
    }
    m3_semantic_output_dispose(&output);
    m3_semantic_test_fixture_dispose(&fixture);
    m3_backend_free(backend);
}

void m3_test_semantic_eos_schedule(m3_test_context *test)
{
    m3_semantic_eos_case(
        test, 2U, 0U, M3_STATUS_OUT_OF_RANGE, 0U);
    m3_semantic_eos_case(
        test, 2U, 1U, M3_STATUS_OUT_OF_RANGE, 0U);
    m3_semantic_eos_case(test, 4U, 3U, M3_STATUS_OK, 2U);
    m3_semantic_eos_case(test, 3U, 3U, M3_STATUS_OK, 2U);
}
