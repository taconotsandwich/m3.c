/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_backend.h"

#include <stdint.h>
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

void m3_test_metal_backend_storage(m3_test_context *test)
{
    const uint8_t source[] = {
        0x10U, 0x21U, 0x32U, 0x43U, 0x54U, 0x65U, 0x76U, 0x87U,
        0x98U, 0xa9U, 0xbaU, 0xcbU, 0xdcU, 0xedU, 0xfeU, 0x0fU
    };
    const uint8_t replacement[] = {0xa1U, 0xb2U, 0xc3U};
    uint8_t downloaded[sizeof(source)] = {0U};
    uint8_t replacement_downloaded[sizeof(replacement)] = {0U};
    m3_backend_info info;
    m3_backend_allocation_stats stats;
    m3_storage *temporary = NULL;
    m3_storage *storage = NULL;
    m3_backend *backend = NULL;
    m3_error error;
    m3_status status;

    status = m3_backend_create_metal(&backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create default Metal backend");
    if (status != M3_STATUS_OK) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_backend_get_info(backend, &info, &error) ==
                       M3_STATUS_OK,
                   "read Metal backend information");
    M3_TEST_EXPECT(test, info.kind == M3_BACKEND_METAL,
                   "backend identifies Metal storage");
    M3_TEST_EXPECT(test, m3_test_device_name_is_sanitized(info.name),
                   "device name contains no control characters");
    M3_TEST_EXPECT(test, info.maximum_storage_bytes >= sizeof(source),
                   "device accepts the tiny test storage");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(backend, &stats, &error) ==
                       M3_STATUS_OK,
                   "read initial Metal allocation statistics");
    M3_TEST_EXPECT(test,
                   stats.live_allocated_bytes == 0U &&
                       stats.live_storage_count == 0U &&
                       stats.peak_allocated_bytes == 0U &&
                       stats.peak_storage_count == 0U,
                   "initial allocation statistics are empty");
    M3_TEST_EXPECT(test,
                   m3_storage_allocate(backend, sizeof(source), 16U,
                                       &storage, &error) == M3_STATUS_OK,
                   "allocate shared Metal storage");
    if (storage == NULL) {
        m3_backend_free(backend);
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_storage_size(storage) == sizeof(source) &&
                       m3_storage_backend(storage) == backend &&
                       m3_storage_data(storage) != NULL,
                   "storage exposes raw size, owner, and shared pointer");
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(backend, &stats, &error) ==
                       M3_STATUS_OK,
                   "read live Metal allocation statistics");
    M3_TEST_EXPECT(test,
                   stats.live_allocated_bytes == sizeof(source) &&
                       stats.peak_allocated_bytes == sizeof(source) &&
                       stats.live_storage_count == 1U &&
                       stats.peak_storage_count == 1U,
                   "allocation statistics track raw storage");
    M3_TEST_EXPECT(test,
                   m3_storage_write(storage, 0U, source, sizeof(source),
                                    &error) == M3_STATUS_OK,
                   "write deterministic Metal bytes");
    M3_TEST_EXPECT(test,
                   m3_storage_read(storage, 0U, downloaded,
                                   sizeof(downloaded), &error) ==
                       M3_STATUS_OK,
                   "read deterministic Metal bytes");
    M3_TEST_EXPECT(test,
                   memcmp(source, downloaded, sizeof(source)) == 0,
                   "shared Metal storage round trip is exact");
    M3_TEST_EXPECT(test,
                   m3_storage_write(storage, 5U, replacement,
                                    sizeof(replacement), &error) ==
                       M3_STATUS_OK,
                   "write a bounded subrange");
    M3_TEST_EXPECT(test,
                   m3_storage_read(storage, 5U, replacement_downloaded,
                                   sizeof(replacement_downloaded), &error) ==
                       M3_STATUS_OK,
                   "read a bounded subrange");
    M3_TEST_EXPECT(test,
                   memcmp(replacement, replacement_downloaded,
                          sizeof(replacement)) == 0,
                   "subrange round trip honors its offset");
    M3_TEST_EXPECT(test,
                   m3_storage_write(storage, sizeof(source), NULL, 0U,
                                    &error) == M3_STATUS_OK,
                   "permit an empty transfer at storage boundary");
    M3_TEST_EXPECT(test,
                   m3_storage_write(storage, sizeof(source), source, 1U,
                                    &error) == M3_STATUS_OUT_OF_RANGE,
                   "reject write beyond storage boundary");
    M3_TEST_EXPECT(test,
                   m3_storage_read(storage, sizeof(source) - 1U,
                                   downloaded, 2U, &error) ==
                       M3_STATUS_OUT_OF_RANGE,
                   "reject read crossing storage boundary");
    M3_TEST_EXPECT(test,
                   m3_storage_write(storage, 0U, NULL, 1U, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject nonempty write from null memory");
    M3_TEST_EXPECT(test,
                   m3_storage_allocate(backend, sizeof(source), 3U,
                                       &temporary, &error) ==
                       M3_STATUS_INVALID_ARGUMENT,
                   "reject invalid storage alignment");
    M3_TEST_EXPECT(test, temporary == NULL,
                   "rejected allocation clears storage output");
    M3_TEST_EXPECT(test,
                   m3_storage_allocate(backend, sizeof(source), 16U,
                                       &temporary, &error) == M3_STATUS_OK,
                   "allocate storage after rejected request");
    m3_storage_free(temporary);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(backend, &stats, &error) ==
                       M3_STATUS_OK,
                   "read post-free Metal statistics");
    M3_TEST_EXPECT(test,
                   stats.live_allocated_bytes == sizeof(source) &&
                       stats.live_storage_count == 1U &&
                       stats.peak_allocated_bytes == sizeof(source) * 2U &&
                       stats.peak_storage_count == 2U,
                   "free updates live statistics and preserves peaks");
    m3_storage_free(storage);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(backend, &stats, &error) ==
                       M3_STATUS_OK,
                   "read final Metal statistics");
    M3_TEST_EXPECT(test,
                   stats.live_allocated_bytes == 0U &&
                       stats.live_storage_count == 0U,
                   "explicit frees clear live statistics");
    m3_backend_free(backend);
}
