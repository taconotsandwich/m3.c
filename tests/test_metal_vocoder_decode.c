/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "vocoder_decode_test.h"

#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t counts[7U];
    size_t call_count;
    uint64_t cancel_at;
    bool totals_valid;
} m3_metal_vocoder_progress;

static bool m3_metal_vocoder_progress_call(
    void *context, uint64_t completed, uint64_t total)
{
    m3_metal_vocoder_progress *progress = context;

    if (progress->call_count < 7U) {
        progress->counts[progress->call_count] = completed;
    }
    ++progress->call_count;
    progress->totals_valid = progress->totals_valid && total == 73U;
    return completed != progress->cancel_at;
}

static void m3_metal_vocoder_progress_init(
    m3_metal_vocoder_progress *progress, uint64_t cancel_at)
{
    (void)memset(progress, 0, sizeof(*progress));
    progress->cancel_at = cancel_at;
    progress->totals_valid = true;
}

static bool m3_metal_vocoder_progress_complete(
    const m3_metal_vocoder_progress *progress)
{
    static const uint64_t expected[] = {0U, 2U, 19U, 36U,
                                        53U, 70U, 73U};

    return progress->totals_valid && progress->call_count == 7U &&
           memcmp(progress->counts, expected, sizeof(expected)) == 0;
}

void test_metal_vocoder_decode_chunk(m3_test_context *test)
{
    const float input_values[] = {0.125F, -0.25F, 0.75F, -0.5F};
    m3_metal_vocoder_progress progress;
    m3_vocoder_runtime *host_runtime = NULL;
    m3_vocoder_runtime *metal_runtime = NULL;
    m3_vocoder_output host_output;
    m3_vocoder_output metal_output;
    m3_backend *host = NULL;
    m3_backend *metal = NULL;
    m3_storage *host_input_storage = NULL;
    m3_storage *metal_input_storage = NULL;
    m3_tensor_view host_latents;
    m3_tensor_view metal_latents;
    m3_backend_allocation_stats before;
    m3_backend_allocation_stats after;
    float host_values[64U];
    float metal_values[64U];
    m3_error error;
    m3_status status;
    size_t index;

    m3_vocoder_output_init(&host_output);
    m3_vocoder_output_init(&metal_output);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&host, &error) == M3_STATUS_OK,
                   "create host decoder oracle backend");
    status = m3_backend_create_metal(&metal, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        m3_backend_free(host);
        M3_TEST_SKIP(test, m3_error_message(&error));
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create real M4 Metal decoder backend");
    if (status != M3_STATUS_OK) {
        m3_backend_free(host);
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_test_runtime(
                       host, &host_runtime, &error) &&
                       m3_vocoder_decode_test_runtime(
                           metal, &metal_runtime, &error) &&
                       m3_vocoder_decode_test_latents(
                           host, 2U, input_values, &host_input_storage,
                           &host_latents, &error) &&
                       m3_vocoder_decode_test_latents(
                           metal, 2U, input_values, &metal_input_storage,
                           &metal_latents, &error),
                   "materialize equal reduced runtimes before stage teardown");
    if (host_runtime == NULL || metal_runtime == NULL ||
        host_input_storage == NULL || metal_input_storage == NULL) {
        m3_storage_free(host_input_storage);
        m3_storage_free(metal_input_storage);
        m3_vocoder_runtime_free(host_runtime);
        m3_vocoder_runtime_free(metal_runtime);
        m3_backend_free(host);
        m3_backend_free(metal);
        return;
    }
    m3_metal_vocoder_progress_init(&progress, 36U);
    (void)m3_backend_get_allocation_stats(metal, &before, &error);
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       metal_runtime, &metal_latents,
                       m3_metal_vocoder_progress_call, &progress,
                       &metal_output, &error) == M3_STATUS_CANCELLED &&
                       progress.call_count == 4U &&
                       metal_output.storage == NULL &&
                       m3_backend_get_allocation_stats(
                           metal, &after, &error) == M3_STATUS_OK &&
                       before.live_allocated_bytes ==
                           after.live_allocated_bytes &&
                       before.live_storage_count == after.live_storage_count,
                   "Metal block-boundary cancellation is allocation atomic");
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       metal_runtime, NULL, NULL, NULL,
                       &metal_output, NULL) == M3_STATUS_INVALID_ARGUMENT &&
                       metal_output.storage == NULL,
                   "Metal public path accepts a null validation error sink");
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       host_runtime, &host_latents, NULL, NULL,
                       &host_output, &error) == M3_STATUS_OK,
                   "execute reduced Host decoder oracle");
    m3_metal_vocoder_progress_init(&progress, UINT64_MAX);
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       metal_runtime, &metal_latents,
                       m3_metal_vocoder_progress_call, &progress,
                       &metal_output, &error) == M3_STATUS_OK &&
                       m3_metal_vocoder_progress_complete(&progress),
                   "Metal preserves all six dependency batches and checkpoints");
    m3_vocoder_runtime_free(host_runtime);
    m3_vocoder_runtime_free(metal_runtime);
    host_runtime = NULL;
    metal_runtime = NULL;
    M3_TEST_EXPECT(test,
                   m3_storage_read(host_output.storage, 0U, host_values,
                                   sizeof(host_values), &error) ==
                           M3_STATUS_OK &&
                       m3_storage_read(metal_output.storage, 0U,
                                       metal_values, sizeof(metal_values),
                                       &error) == M3_STATUS_OK,
                   "owned Host and Metal waveforms outlive their runtimes");
    for (index = 0U; index < 64U; ++index) {
        M3_TEST_EXPECT_F32(
            test, metal_values[index], host_values[index],
            0x1p-14F, 0x1p-14F,
            "real Metal 73-op waveform matches the Host oracle");
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_decode_chunk(
                       NULL, &metal_latents, NULL, NULL,
                       &metal_output, NULL) == M3_STATUS_INVALID_ARGUMENT &&
                       metal_output.storage != NULL,
                   "failed null-error retry preserves published Metal output");
    m3_vocoder_output_dispose(&host_output);
    m3_vocoder_output_dispose(&metal_output);
    m3_storage_free(host_input_storage);
    m3_storage_free(metal_input_storage);
    m3_backend_free(host);
    m3_backend_free(metal);
}
