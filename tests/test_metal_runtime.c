/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static bool m3_test_device_name_is_sanitized(const char *name)
{
    size_t index;

    if (name == NULL || name[0] == '\0') {
        return false;
    }
    for (index = 0U; name[index] != '\0'; ++index) {
        unsigned char value = (unsigned char)name[index];

        if (value < 0x20U || value == 0x7fU) {
            return false;
        }
    }
    return true;
}

void m3_test_metal_runtime_shared_buffer(m3_test_context *test)
{
    const uint64_t shape[] = {4U};
    const uint8_t source[] = {
        0x10U, 0x21U, 0x32U, 0x43U, 0x54U, 0x65U, 0x76U, 0x87U,
        0x98U, 0xa9U, 0xbaU, 0xcbU, 0xdcU, 0xedU, 0xfeU, 0x0fU
    };
    const uint8_t replacement[] = {0xa1U, 0xb2U, 0xc3U};
    uint8_t downloaded[sizeof(source)] = {0U};
    uint8_t replacement_downloaded[sizeof(replacement)] = {0U};
    m3_tensor_metadata metadata;
    m3_tensor_metadata invalid_metadata;
    m3_gpu_device_info info;
    m3_gpu_allocation_stats stats;
    const m3_tensor_metadata *buffer_metadata;
    m3_gpu_buffer *temporary_buffer = NULL;
    m3_gpu_buffer *buffer = NULL;
    m3_gpu *gpu = NULL;
    m3_error error;
    m3_status status;

    status = m3_gpu_create(&gpu, &error);
    if (status == M3_STATUS_UNSUPPORTED &&
        strstr(m3_error_message(&error), "Metal unavailable:") != NULL) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create default Metal runtime");
    if (status != M3_STATUS_OK) {
        return;
    }

    M3_TEST_EXPECT(test,
                   m3_gpu_get_device_info(gpu, &info, &error) ==
                       M3_STATUS_OK,
                   "read sanitized Metal device information");
    M3_TEST_EXPECT(test, m3_test_device_name_is_sanitized(info.name),
                   "device name contains no control characters");
    M3_TEST_EXPECT(test, info.maximum_buffer_bytes >= sizeof(source),
                   "device accepts the tiny test buffer");

    M3_TEST_EXPECT(test,
                   m3_gpu_get_allocation_stats(gpu, &stats, &error) ==
                       M3_STATUS_OK,
                   "read initial Metal allocation statistics");
    M3_TEST_EXPECT(test,
                   stats.live_allocated_bytes == 0U &&
                       stats.live_buffer_count == 0U &&
                       stats.peak_allocated_bytes == 0U &&
                       stats.peak_buffer_count == 0U,
                   "initial allocation statistics are empty");

    M3_TEST_EXPECT(test,
                   m3_tensor_metadata_init(&metadata, M3_DTYPE_F32, 1U,
                                           shape, &error) == M3_STATUS_OK,
                   "initialize tiny tensor metadata");
    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_create(gpu, &metadata, &buffer, &error) ==
                       M3_STATUS_OK,
                   "allocate shared Metal tensor buffer");
    if (buffer == NULL) {
        m3_gpu_free(gpu);
        return;
    }

    buffer_metadata = m3_gpu_buffer_metadata(buffer);
    M3_TEST_EXPECT(test,
                   buffer_metadata != NULL &&
                       buffer_metadata->dtype == M3_DTYPE_F32 &&
                       buffer_metadata->element_count == 4U &&
                       buffer_metadata->byte_count == sizeof(source),
                   "buffer preserves validated tensor metadata");
    M3_TEST_EXPECT(test,
                   m3_gpu_get_allocation_stats(gpu, &stats, &error) ==
                       M3_STATUS_OK,
                   "read live Metal allocation statistics");
    M3_TEST_EXPECT(test,
                   stats.live_allocated_bytes == sizeof(source) &&
                       stats.peak_allocated_bytes == sizeof(source) &&
                       stats.live_buffer_count == 1U &&
                       stats.peak_buffer_count == 1U,
                   "allocation statistics track the shared buffer");

    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_upload(buffer, 0U, source,
                                        sizeof(source), &error) ==
                       M3_STATUS_OK,
                   "upload deterministic bytes");
    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_download(buffer, 0U, downloaded,
                                          sizeof(downloaded), &error) ==
                       M3_STATUS_OK,
                   "download deterministic bytes");
    M3_TEST_EXPECT(test,
                   memcmp(source, downloaded, sizeof(source)) == 0,
                   "shared Metal buffer round trip is exact");
    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_upload(buffer, 5U, replacement,
                                        sizeof(replacement), &error) ==
                       M3_STATUS_OK,
                   "upload a bounded subrange");
    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_download(buffer, 5U,
                                          replacement_downloaded,
                                          sizeof(replacement_downloaded),
                                          &error) == M3_STATUS_OK,
                   "download a bounded subrange");
    M3_TEST_EXPECT(test,
                   memcmp(replacement, replacement_downloaded,
                          sizeof(replacement)) == 0,
                   "subrange round trip honors its offset");

    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_upload(buffer, sizeof(source), NULL, 0U,
                                        &error) == M3_STATUS_OK,
                   "permit an empty transfer at the buffer boundary");
    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_upload(buffer, sizeof(source), source, 1U,
                                        &error) == M3_STATUS_OUT_OF_RANGE,
                   "reject upload beyond the buffer boundary");
    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_download(buffer, sizeof(source) - 1U,
                                          downloaded, 2U, &error) ==
                       M3_STATUS_OUT_OF_RANGE,
                   "reject download crossing the buffer boundary");
    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_upload(buffer, 0U, NULL, 1U, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject nonempty upload from null memory");

    invalid_metadata = metadata;
    invalid_metadata.byte_count += 1U;
    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_create(gpu, &invalid_metadata,
                                        &temporary_buffer, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject forged tensor metadata counts");
    M3_TEST_EXPECT(test, temporary_buffer == NULL,
                   "rejected allocation clears buffer output");

    M3_TEST_EXPECT(test,
                   m3_gpu_buffer_create(gpu, &metadata, &temporary_buffer,
                                        &error) == M3_STATUS_OK,
                   "allocate buffer after rejected metadata");
    m3_gpu_buffer_free(temporary_buffer);
    M3_TEST_EXPECT(test,
                   m3_gpu_get_allocation_stats(gpu, &stats, &error) ==
                       M3_STATUS_OK,
                   "read post-free Metal allocation statistics");
    M3_TEST_EXPECT(test,
                   stats.live_allocated_bytes == sizeof(source) &&
                       stats.live_buffer_count == 1U &&
                       stats.peak_allocated_bytes == sizeof(source) * 2U &&
                       stats.peak_buffer_count == 2U,
                   "free updates live statistics and preserves peaks");

    m3_gpu_buffer_free(buffer);
    M3_TEST_EXPECT(test,
                   m3_gpu_get_allocation_stats(gpu, &stats, &error) ==
                       M3_STATUS_OK,
                   "read final Metal allocation statistics");
    M3_TEST_EXPECT(test,
                   stats.live_allocated_bytes == 0U &&
                       stats.live_buffer_count == 0U &&
                       stats.peak_allocated_bytes == sizeof(source) * 2U &&
                       stats.peak_buffer_count == 2U,
                   "all explicit frees clear live statistics");

    m3_gpu_free(gpu);
}
