/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "waveform_test.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t cancel_at;
    uint64_t expected_total;
    uint64_t last;
    size_t call_count;
    bool valid;
} m3_waveform_progress;

static void m3_waveform_progress_init(
    m3_waveform_progress *progress, uint64_t total, uint64_t cancel_at)
{
    (void)memset(progress, 0, sizeof(*progress));
    progress->cancel_at = cancel_at;
    progress->expected_total = total;
    progress->valid = true;
}

static bool m3_waveform_progress_call(
    void *context, uint64_t completed, uint64_t total)
{
    m3_waveform_progress *progress = context;

    progress->valid = progress->valid &&
                      total == progress->expected_total &&
                      completed == (uint64_t)progress->call_count;
    progress->last = completed;
    ++progress->call_count;
    return completed != progress->cancel_at;
}

void test_waveform_assembly_boundaries(m3_test_context *test)
{
    static const struct {
        uint64_t frames;
        size_t chunks;
        size_t samples;
    } cases[] = {
        {1U, 1U, 1536U},
        {100U, 1U, 176128U},
        {101U, 1U, 177664U},
        {200U, 1U, 352768U},
        {201U, 2U, 354304U},
        {300U, 2U, 529408U},
        {301U, 3U, 530944U}
    };
    m3_backend *backend = NULL;
    m3_error error;
    size_t index;

    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK,
                   "create waveform boundary backend");
    if (backend == NULL) {
        return;
    }
    for (index = 0U; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        m3_waveform_test_fixture fixture;
        m3_waveform_progress progress;
        m3_vocoder_output output;
        uint64_t total = (uint64_t)cases[index].chunks * 2U;
        bool created;

        m3_vocoder_output_init(&output);
        created = m3_waveform_test_fixture_create(
            &fixture, backend, cases[index].frames, &error);
        m3_waveform_progress_init(&progress, total, UINT64_MAX);
        M3_TEST_EXPECT(
            test,
            created && fixture.chunk_count == cases[index].chunks &&
                m3_waveform_test_output_samples(cases[index].frames) ==
                    cases[index].samples &&
                m3_waveform_assemble(
                    fixture.chunks, fixture.chunk_count,
                    cases[index].frames, m3_waveform_progress_call,
                    &progress, &output, &error) == M3_STATUS_OK &&
                progress.valid && progress.last == total &&
                progress.call_count == (size_t)total + 1U &&
                m3_waveform_test_output_matches(
                    &fixture, &output, &error),
            "assemble exact official boundary crop with independent oracle");
        m3_vocoder_output_dispose(&output);
        m3_waveform_test_fixture_dispose(&fixture);
    }
    m3_backend_free(backend);
}

void test_waveform_assembly_clamp_bits(m3_test_context *test)
{
    static const uint32_t input[] = {
        UINT32_C(0x40000000), UINT32_C(0xc0000000),
        UINT32_C(0x3f800000), UINT32_C(0xbf800000),
        UINT32_C(0x00000000), UINT32_C(0x80000000),
        UINT32_C(0x00000001), UINT32_C(0x80000001),
        UINT32_C(0x7fc12345), UINT32_C(0xffc54321),
        UINT32_C(0x7f800000), UINT32_C(0xff800000),
        UINT32_C(0x7fa12345), UINT32_C(0xffa54321)
    };
    static const uint32_t expected[] = {
        UINT32_C(0x3f800000), UINT32_C(0xbf800000),
        UINT32_C(0x3f800000), UINT32_C(0xbf800000),
        UINT32_C(0x00000000), UINT32_C(0x80000000),
        UINT32_C(0x00000001), UINT32_C(0x80000001),
        UINT32_C(0x7fc12345), UINT32_C(0xffc54321),
        UINT32_C(0x3f800000), UINT32_C(0xbf800000),
        UINT32_C(0x7fa12345), UINT32_C(0xffa54321)
    };
    m3_waveform_test_fixture fixture;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    uint32_t actual[sizeof(input) / sizeof(input[0])] = {0U};
    m3_error error;
    bool created;

    m3_waveform_test_fixture_init(&fixture);
    m3_vocoder_output_init(&output);
    created = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
              m3_waveform_test_fixture_create(
                  &fixture, backend, 1U, &error);
    M3_TEST_EXPECT(test, created, "create waveform clamp fixture");
    if (!created) {
        m3_backend_free(backend);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_storage_write(
            fixture.storages[0], 0U, input, sizeof(input), &error) ==
                M3_STATUS_OK &&
            m3_waveform_assemble(
                fixture.chunks, fixture.chunk_count, 1U, NULL, NULL,
                &output, &error) == M3_STATUS_OK &&
            m3_storage_read(
                output.storage, 0U, actual, sizeof(actual), &error) ==
                M3_STATUS_OK &&
            memcmp(actual, expected, sizeof(expected)) == 0,
        "clamp only out-of-range samples and preserve exact special bits");
    m3_vocoder_output_dispose(&output);
    m3_waveform_test_fixture_dispose(&fixture);
    m3_backend_free(backend);
}

void test_waveform_assembly_alias_lifetime(m3_test_context *test)
{
    m3_waveform_test_fixture fixture;
    m3_vocoder_output output;
    m3_backend *backend = NULL;
    m3_storage *aliased = NULL;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    uint32_t first = 0U;
    m3_error error;
    bool created;

    m3_waveform_test_fixture_init(&fixture);
    m3_vocoder_output_init(&output);
    created = m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
              m3_waveform_test_fixture_create(
                  &fixture, backend, 1U, &error) &&
              m3_backend_get_allocation_stats(
                  backend, &before, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, created, "create waveform alias fixture");
    if (!created) {
        m3_waveform_test_fixture_dispose(&fixture);
        m3_backend_free(backend);
        return;
    }
    aliased = fixture.storages[0];
    output.storage = aliased;
    output.waveform = fixture.chunks[0];
    fixture.storages[0] = NULL;
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble(
            &output.waveform, 1U, 1U, NULL, NULL, &output, &error) ==
                M3_STATUS_OK &&
            output.storage != aliased &&
            m3_waveform_test_output_matches(&fixture, &output, &error) &&
            m3_backend_get_allocation_stats(
                backend, &after, &error) == M3_STATUS_OK &&
            before.live_allocated_bytes == after.live_allocated_bytes &&
            before.live_storage_count == after.live_storage_count,
        "finish all aliased reads before atomically replacing old output");
    m3_waveform_test_fixture_dispose(&fixture);
    M3_TEST_EXPECT(
        test,
        m3_storage_read(output.storage, 0U, &first, sizeof(first), &error) ==
                M3_STATUS_OK &&
            first == m3_waveform_test_pattern(0U, 0U, 0U),
        "assembled output remains owned after all input views are released");
    m3_vocoder_output_dispose(&output);
    m3_backend_free(backend);
}
