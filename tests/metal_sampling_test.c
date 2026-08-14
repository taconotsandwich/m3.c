/* SPDX-License-Identifier: GPL-2.0-only */

#include "metal_sampling_test.h"

#include <stdint.h>
#include <string.h>

bool m3_metal_sampling_fixture_init(
    m3_test_context *test, m3_metal_sampling_fixture *fixture)
{
    m3_error error;
    m3_status status;

    (void)memset(fixture, 0, sizeof(*fixture));
    if (!m3_op_test_fixture_init(&fixture->host)) {
        M3_TEST_EXPECT(test, false, "create sampling host fixture");
        return false;
    }
    status = m3_backend_create_metal(&fixture->metal.backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        m3_op_test_fixture_dispose(&fixture->host);
        return false;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create sampling Metal fixture");
    if (status != M3_STATUS_OK) {
        m3_op_test_fixture_dispose(&fixture->host);
        return false;
    }
    return true;
}

void m3_metal_sampling_fixture_dispose(
    m3_metal_sampling_fixture *fixture)
{
    m3_op_test_fixture_dispose(&fixture->metal);
    m3_op_test_fixture_dispose(&fixture->host);
}

bool m3_metal_sampling_tensor(
    m3_metal_sampling_fixture *fixture, m3_metal_sampling_view *view,
    m3_dtype dtype, uint8_t rank, const uint64_t *shape,
    const void *values)
{
    return m3_op_test_tensor(&fixture->host, &view->host, dtype, rank,
                             shape, values) &&
           m3_op_test_tensor(&fixture->metal, &view->metal, dtype, rank,
                             shape, values);
}

static bool m3_metal_sampling_strided_one(
    m3_op_test_fixture *fixture, m3_tensor_view *view, m3_dtype dtype,
    uint8_t rank, const uint64_t *shape, const size_t *strides,
    size_t byte_offset, size_t byte_count, const void *values)
{
    m3_storage *storage = NULL;
    m3_error error;

    m3_tensor_view_init(view);
    return m3_op_test_storage(fixture, byte_count, &storage) &&
           m3_storage_write(storage, 0U, values, byte_count, &error) ==
               M3_STATUS_OK &&
           m3_tensor_view_strided(view, storage, dtype, rank, shape,
                                  strides, byte_offset, &error) ==
               M3_STATUS_OK;
}

bool m3_metal_sampling_strided(
    m3_metal_sampling_fixture *fixture, m3_metal_sampling_view *view,
    m3_dtype dtype, uint8_t rank, const uint64_t *shape,
    const size_t *strides, size_t byte_offset, size_t byte_count,
    const void *values)
{
    return m3_metal_sampling_strided_one(
               &fixture->host, &view->host, dtype, rank, shape, strides,
               byte_offset, byte_count, values) &&
           m3_metal_sampling_strided_one(
               &fixture->metal, &view->metal, dtype, rank, shape, strides,
               byte_offset, byte_count, values);
}

bool m3_metal_sampling_execute(
    m3_metal_sampling_fixture *fixture, const m3_command *host_commands,
    const m3_command *metal_commands, size_t command_count,
    m3_error *error)
{
    return m3_op_test_execute(&fixture->host, host_commands, command_count,
                              NULL, error) == M3_STATUS_OK &&
           m3_op_test_execute(&fixture->metal, metal_commands, command_count,
                              NULL, error) == M3_STATUS_OK;
}

bool m3_metal_sampling_equal(const m3_metal_sampling_view *view)
{
    const uint8_t *host;
    const uint8_t *metal;

    if (view->host.metadata.byte_count != view->metal.metadata.byte_count) {
        return false;
    }
    if (view->host.metadata.byte_count == 0U) {
        return true;
    }
    host = m3_storage_const_data(view->host.storage);
    metal = m3_storage_const_data(view->metal.storage);
    return memcmp(host + view->host.byte_offset,
                  metal + view->metal.byte_offset,
                  view->host.metadata.byte_count) == 0;
}
