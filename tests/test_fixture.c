/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

static bool m3_test_read_fixture(const char *path, uint8_t *data, size_t size)
{
    size_t offset = 0U;
    int descriptor = open(path, O_RDONLY);

    if (descriptor < 0) {
        return false;
    }
    while (offset < size) {
        ssize_t count = read(descriptor, data + offset, size - offset);

        if (count <= 0) {
            (void)close(descriptor);
            return false;
        }
        offset += (size_t)count;
    }
    if (close(descriptor) != 0) {
        return false;
    }
    return true;
}

void m3_test_fixture_contract(m3_test_context *test)
{
    static const uint8_t fixture_data[] = {
        0x00U, 0x7bU, 0x22U, 0x78U, 0x22U, 0x3aU, 0x31U, 0x7dU, 0xffU
    };
    static const m3_test_fixture fixture = {
        "binary header with sentinel bytes",
        fixture_data,
        sizeof(fixture_data)
    };
    static const m3_test_fixture invalid_fixture = {
        "invalid missing data",
        NULL,
        1U
    };
    uint8_t actual[sizeof(fixture_data)] = {0U};
    m3_test_temp_file file = {{0}};

    M3_TEST_EXPECT(test, m3_test_temp_file_create(&file, &fixture),
                   "create deterministic binary fixture file");
    M3_TEST_EXPECT(test, file.path[0] != '\0',
                   "fixture exposes temporary path");
    M3_TEST_EXPECT(test,
                   m3_test_read_fixture(file.path, actual, sizeof(actual)),
                   "read complete binary fixture");
    M3_TEST_EXPECT(test,
                   memcmp(actual, fixture.data, fixture.size) == 0,
                   "temporary file preserves fixture bytes");
    M3_TEST_EXPECT(test, m3_test_temp_file_remove(&file),
                   "remove temporary fixture file");
    M3_TEST_EXPECT(test, file.path[0] == '\0',
                   "fixture removal clears temporary path");
    M3_TEST_EXPECT(test, !m3_test_temp_file_remove(&file),
                   "reject duplicate fixture removal");
    M3_TEST_EXPECT(test,
                   !m3_test_temp_file_create(&file, &invalid_fixture),
                   "reject fixture with missing bytes");
    M3_TEST_EXPECT(test, file.path[0] == '\0',
                   "rejected fixture leaves no temporary path");

    M3_TEST_EXPECT_F32(test, 1.0005F, 1.0F, 0.001F, 0.0F,
                       "absolute F32 tolerance");
    M3_TEST_EXPECT_F32(test, 1001.0F, 1000.0F, 0.0F, 0.001F,
                       "relative F32 tolerance");
    M3_TEST_EXPECT(test,
                   !m3_test_f32_close(1.01F, 1.0F, 0.001F, 0.001F),
                   "reject value outside F32 tolerances");
    M3_TEST_EXPECT(test,
                   !m3_test_f32_close(NAN, 1.0F, 0.001F, 0.001F),
                   "reject NaN numeric fixture");
    M3_TEST_EXPECT(test,
                   !m3_test_f32_close(1.0F, 1.0F, -1.0F, 0.0F),
                   "reject negative F32 tolerance");
    M3_TEST_EXPECT(test,
                   m3_test_f32_close(INFINITY, INFINITY, 0.0F, 0.0F),
                   "accept identical infinity fixture");
}
