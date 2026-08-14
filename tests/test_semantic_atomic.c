/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "semantic_runtime_test.h"

#include "m3_backend.h"

#include <stdint.h>
#include <string.h>

static bool m3_semantic_atomic_preserved(
    const m3_semantic_output *output, const m3_storage *storage,
    uint16_t sentinel, const m3_rng *rng, const m3_rng *initial)
{
    uint16_t actual = 0U;

    return output->storage == storage &&
           m3_semantic_test_output_bits(output, &actual, 1U) &&
           actual == sentinel &&
           memcmp(rng, initial, sizeof(*rng)) == 0;
}

static void m3_semantic_fault_case(
    m3_test_context *test, m3_semantic_test_failure failure,
    bool null_error)
{
    const uint16_t sentinel = UINT16_C(0x5a5a);
    m3_semantic_test_progress progress = {
        .cancel_at = UINT64_MAX
    };
    m3_semantic_test_fixture fixture;
    m3_semantic_output output;
    m3_storage *original_storage;
    m3_backend *backend = NULL;
    m3_rng initial;
    m3_rng rng;
    m3_error error;
    bool ready;
    m3_status status;

    m3_error_reset(&error);
    ready = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
            m3_semantic_test_fixture_init(&fixture, backend) &&
            m3_rng_seed(&rng, UINT64_C(0xa1b2c3d4),
                        UINT64_C(0x55667788), &error) == M3_STATUS_OK &&
            m3_semantic_test_seed_output(
                backend, sentinel, &output);
    M3_TEST_EXPECT(test, ready, "create semantic atomic fault fixture");
    if (!ready) {
        m3_backend_free(backend);
        return;
    }
    m3_semantic_test_outcomes(&fixture, 2U, SIZE_MAX);
    fixture.failure = failure;
    initial = rng;
    original_storage = output.storage;
    status = m3_semantic_generate_core(
        &fixture.operations, &fixture.prompt, 1U, &rng,
        m3_semantic_test_progress_callback, &progress, &output,
        null_error ? NULL : &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INTERNAL,
                   "surface injected semantic phase failure");
    M3_TEST_EXPECT(
        test,
        m3_semantic_atomic_preserved(
            &output, original_storage, sentinel, &rng, &initial),
        "phase failure preserves caller output and RNG exactly");
    M3_TEST_EXPECT(
        test, fixture.finish_calls == 1U,
        "phase failure releases the operation component exactly once");
    m3_semantic_output_dispose(&output);
    m3_semantic_test_fixture_dispose(&fixture);
    m3_backend_free(backend);
}

static void m3_semantic_cancel_case(m3_test_context *test,
                                    uint64_t checkpoint)
{
    const uint16_t sentinel = UINT16_C(0x6b6b);
    m3_semantic_test_progress progress = {
        .cancel_at = checkpoint
    };
    m3_semantic_test_fixture fixture;
    m3_semantic_output output;
    m3_storage *original_storage;
    m3_backend *backend = NULL;
    m3_rng initial;
    m3_rng rng;
    m3_error error;
    bool ready;
    m3_status status;

    m3_error_reset(&error);
    ready = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
            m3_semantic_test_fixture_init(&fixture, backend) &&
            m3_rng_seed(&rng, UINT64_C(0x10203040),
                        UINT64_C(0x50607080), &error) == M3_STATUS_OK &&
            m3_semantic_test_seed_output(
                backend, sentinel, &output);
    M3_TEST_EXPECT(test, ready,
                   "create semantic cancellation fixture");
    if (!ready) {
        m3_backend_free(backend);
        return;
    }
    m3_semantic_test_outcomes(&fixture, 2U, SIZE_MAX);
    initial = rng;
    original_storage = output.storage;
    status = m3_semantic_generate_core(
        &fixture.operations, &fixture.prompt, 1U, &rng,
        m3_semantic_test_progress_callback, &progress, &output, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_CANCELLED,
                   "cancel semantic generation at an exact checkpoint");
    M3_TEST_EXPECT(
        test,
        m3_semantic_atomic_preserved(
            &output, original_storage, sentinel, &rng, &initial),
        "cancellation preserves caller output and RNG exactly");
    M3_TEST_EXPECT(
        test,
        progress.count != 0U &&
            progress.values[progress.count - 1U] == checkpoint &&
            fixture.finish_calls == 1U,
        "cancellation is observed at the requested stable checkpoint");
    m3_semantic_output_dispose(&output);
    m3_semantic_test_fixture_dispose(&fixture);
    m3_backend_free(backend);
}

void m3_test_semantic_atomic_faults(m3_test_context *test)
{
    static const m3_semantic_test_failure failures[] = {
        M3_SEMANTIC_TEST_FAIL_START,
        M3_SEMANTIC_TEST_FAIL_PREFILL,
        M3_SEMANTIC_TEST_FAIL_EMBEDDING,
        M3_SEMANTIC_TEST_FAIL_DECODE,
        M3_SEMANTIC_TEST_FAIL_FEEDBACK,
        M3_SEMANTIC_TEST_FAIL_ADVANCE
    };
    size_t index;

    for (index = 0U; index < sizeof(failures) / sizeof(failures[0]);
         ++index) {
        m3_semantic_fault_case(test, failures[index], index == 2U);
    }
}

void m3_test_semantic_cancellation(m3_test_context *test)
{
    static const uint64_t checkpoints[] = {
        0U, 1U, 40U, 41U, 48U, 49U, 50U, 97U
    };
    size_t index;

    for (index = 0U;
         index < sizeof(checkpoints) / sizeof(checkpoints[0]); ++index) {
        m3_semantic_cancel_case(test, checkpoints[index]);
    }
}
