/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "waveform_test.h"

#include <stdlib.h>
#include <string.h>

void test_metal_waveform_assembly(m3_test_context *test)
{
    m3_waveform_test_fixture host_fixture;
    m3_waveform_test_fixture metal_fixture;
    m3_vocoder_output host_output;
    m3_vocoder_output metal_output;
    m3_backend *host = NULL;
    m3_backend *metal = NULL;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    uint32_t *host_bits = NULL;
    uint32_t *metal_bits = NULL;
    size_t output_samples = m3_waveform_test_output_samples(201U);
    size_t output_bytes = output_samples * 2U * sizeof(uint32_t);
    m3_error error;
    m3_status status;
    bool created;

    m3_waveform_test_fixture_init(&host_fixture);
    m3_waveform_test_fixture_init(&metal_fixture);
    m3_vocoder_output_init(&host_output);
    m3_vocoder_output_init(&metal_output);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&host, &error) == M3_STATUS_OK,
                   "create Host waveform oracle backend");
    status = m3_backend_create_metal(&metal, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        m3_backend_free(host);
        M3_TEST_SKIP(test, m3_error_message(&error));
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create real M4 Metal waveform backend");
    if (status != M3_STATUS_OK) {
        m3_backend_free(host);
        return;
    }
    created = m3_waveform_test_fixture_create(
                  &host_fixture, host, 201U, &error) &&
              m3_waveform_test_fixture_create(
                  &metal_fixture, metal, 201U, &error) &&
              m3_backend_get_allocation_stats(
                  metal, &before, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, created,
                   "create equal Host and Metal cropped chunk fixtures");
    if (!created) {
        m3_waveform_test_fixture_dispose(&host_fixture);
        m3_waveform_test_fixture_dispose(&metal_fixture);
        m3_backend_free(host);
        m3_backend_free(metal);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_waveform_assemble(
            host_fixture.chunks, host_fixture.chunk_count, 201U, NULL,
            NULL, &host_output, &error) == M3_STATUS_OK &&
            m3_waveform_assemble(
                metal_fixture.chunks, metal_fixture.chunk_count, 201U,
                NULL, NULL, &metal_output, &error) == M3_STATUS_OK &&
            m3_waveform_test_output_matches(
                &host_fixture, &host_output, &error) &&
            m3_waveform_test_output_matches(
                &metal_fixture, &metal_output, &error) &&
            m3_backend_get_allocation_stats(
                metal, &after, &error) == M3_STATUS_OK &&
            after.live_allocated_bytes ==
                before.live_allocated_bytes + output_bytes &&
            after.live_storage_count == before.live_storage_count + 1U,
        "real Metal storage matches the independent 201-frame crop oracle");
    m3_waveform_test_fixture_dispose(&host_fixture);
    m3_waveform_test_fixture_dispose(&metal_fixture);
    host_bits = malloc(output_bytes);
    metal_bits = malloc(output_bytes);
    M3_TEST_EXPECT(
        test,
        host_bits != NULL && metal_bits != NULL &&
            m3_storage_read(
                host_output.storage, 0U, host_bits, output_bytes,
                &error) == M3_STATUS_OK &&
            m3_storage_read(
                metal_output.storage, 0U, metal_bits, output_bytes,
                &error) == M3_STATUS_OK &&
            memcmp(host_bits, metal_bits, output_bytes) == 0,
        "owned Host and Metal assemblies outlive all decoded chunk inputs");
    free(host_bits);
    free(metal_bits);
    m3_vocoder_output_dispose(&host_output);
    m3_vocoder_output_dispose(&metal_output);
    m3_backend_free(host);
    m3_backend_free(metal);
}
