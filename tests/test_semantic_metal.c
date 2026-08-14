/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "semantic_runtime_test.h"

#include "m3_backend.h"

#include <stdint.h>
#include <string.h>

static bool m3_semantic_metal_bit(
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

void m3_test_semantic_reduced_metal(m3_test_context *test)
{
    m3_semantic_test_progress progress = {
        .cancel_at = UINT64_MAX
    };
    m3_semantic_test_fixture fixture;
    m3_semantic_output output;
    m3_backend_allocation_stats stats;
    m3_backend *backend = NULL;
    m3_rng expected;
    m3_rng initial;
    m3_rng rng;
    m3_error error;
    m3_status status;
    bool ready;

    m3_error_reset(&error);
    status = m3_backend_create_metal(&backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create reduced semantic Metal backend");
    if (status != M3_STATUS_OK) {
        return;
    }
    ready = m3_semantic_test_fixture_init(&fixture, backend) &&
            m3_rng_seed(&rng, UINT64_C(0x31415926),
                        UINT64_C(0x27182818), &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "create reduced semantic Metal fixture");
    if (!ready) {
        m3_backend_free(backend);
        return;
    }
    m3_semantic_test_outcomes(&fixture, 3U, SIZE_MAX);
    initial = rng;
    m3_semantic_output_init(&output);
    status = m3_semantic_generate_core(
        &fixture.operations, &fixture.prompt, 2U, &rng,
        m3_semantic_test_progress_callback, &progress, &output, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "run reduced semantic scheduler through real Metal");
    M3_TEST_EXPECT(
        test,
        fixture.decode_calls == 3U && fixture.feedback_calls == 2U &&
            fixture.advance_calls == 2U && fixture.contract_valid &&
            output.frame_hiddens.metadata.shape[1] == 2U &&
            m3_semantic_metal_bit(
                &output, 0U, 0U, UINT16_C(0x3f81)) &&
            m3_semantic_metal_bit(
                &output, 0U, 7U, UINT16_C(0x4217)) &&
            m3_semantic_metal_bit(
                &output, 1U, 0U, UINT16_C(0x3f82)) &&
            m3_semantic_metal_bit(
                &output, 1U, 7U, UINT16_C(0x4227)),
        "Metal preserves warmup discard, frame order, and eight slots");
    M3_TEST_EXPECT(
        test,
        m3_semantic_test_expected_rng(&initial, 24U, &expected) &&
            memcmp(&expected, &rng, sizeof(rng)) == 0 &&
            progress.total_stable && progress.total == 146U &&
            progress.count == 147U,
        "Metal shares exact RNG and progress scheduler semantics");
    m3_semantic_output_dispose(&output);
    m3_semantic_test_fixture_dispose(&fixture);
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 0U &&
            stats.live_storage_count == 0U,
        "reduced Metal generation releases every owned storage");
    m3_backend_free(backend);
}
