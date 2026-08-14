/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "weight_stage_fixture.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    size_t open_calls;
    size_t close_calls;
    bool pass_through;
} m3_stage_open_probe;

static int m3_stage_probe_open(void *context_pointer, const char *path,
                               int flags)
{
    m3_stage_open_probe *probe = context_pointer;

    ++probe->open_calls;
    if (probe->pass_through) {
        return open(path, flags);
    }
    errno = EACCES;
    return -1;
}

static int m3_stage_probe_close(void *context_pointer, int descriptor)
{
    m3_stage_open_probe *probe = context_pointer;

    ++probe->close_calls;
    return close(descriptor);
}

static bool m3_stage_storage_limit_case(uint64_t maximum_storage_bytes)
{
    m3_weight_stage_test_fixture fixture;
    m3_weight_stage_fake_context *fake = NULL;
    m3_stage_open_probe probe = {0};
    m3_backend_allocation_stats stats;
    m3_weight_stage_io io;
    m3_weight_stage stage;
    m3_backend *backend = NULL;
    m3_error error;
    bool success;

    m3_weight_stage_init(&stage);
    if (!m3_weight_stage_test_fixture_create(&fixture, &error) ||
        !m3_weight_stage_test_fake_backend_create(
            maximum_storage_bytes, 0U, SIZE_MAX, &backend, &fake, &error)) {
        m3_backend_free(backend);
        (void)m3_weight_stage_test_fixture_dispose(&fixture);
        return false;
    }
    m3_weight_stage_io_init(&io);
    io.context = &probe;
    io.open_file = m3_stage_probe_open;
    success = m3_weight_stage_load_with_io(
                  &stage, &fixture.table, backend, NULL, NULL, &io,
                  &error) == M3_STATUS_OUT_OF_RANGE &&
              probe.open_calls == 0U && fake->allocation_calls == 0U &&
              m3_backend_get_allocation_stats(backend, &stats, &error) ==
                  M3_STATUS_OK &&
              stats.live_allocated_bytes == 0U &&
              stats.live_storage_count == 0U;
    m3_weight_stage_dispose(&stage);
    m3_backend_free(backend);
    if (!m3_weight_stage_test_fixture_dispose(&fixture)) {
        success = false;
    }
    return success;
}

static bool m3_stage_working_set_case(void)
{
    m3_weight_stage_test_fixture fixture;
    m3_weight_stage_fake_context *fake = NULL;
    m3_stage_open_probe probe = {0};
    m3_backend_allocation_stats stats;
    m3_weight_stage_io io;
    m3_weight_stage stage;
    m3_weight_stage preserved;
    m3_backend *backend = NULL;
    m3_error error;
    bool success;

    m3_weight_stage_init(&stage);
    if (!m3_weight_stage_test_fixture_create(&fixture, &error) ||
        !m3_weight_stage_test_fake_backend_create(
            64U, 43U, SIZE_MAX, &backend, &fake, &error) ||
        m3_weight_stage_load(&stage, &fixture.table, backend, NULL, NULL,
                             &error) != M3_STATUS_OK) {
        m3_weight_stage_dispose(&stage);
        m3_backend_free(backend);
        (void)m3_weight_stage_test_fixture_dispose(&fixture);
        return false;
    }
    preserved = stage;
    m3_weight_stage_io_init(&io);
    io.context = &probe;
    io.open_file = m3_stage_probe_open;
    success = m3_weight_stage_load_with_io(
                  &stage, &fixture.table, backend, NULL, NULL, &io, NULL) ==
                  M3_STATUS_OUT_OF_MEMORY &&
              memcmp(&stage, &preserved, sizeof(stage)) == 0 &&
              probe.open_calls == 0U && fake->allocation_calls == 2U &&
              m3_backend_get_allocation_stats(backend, &stats, &error) ==
                  M3_STATUS_OK &&
              stats.live_allocated_bytes == 22U &&
              stats.live_storage_count == 2U &&
              stats.peak_allocated_bytes == 22U &&
              stats.peak_storage_count == 2U;
    m3_weight_stage_dispose(&stage);
    success = success && fake->free_calls == 2U;
    m3_backend_free(backend);
    if (!m3_weight_stage_test_fixture_dispose(&fixture)) {
        success = false;
    }
    return success;
}

static bool m3_stage_allocation_failure_case(void)
{
    m3_weight_stage_test_fixture fixture;
    m3_weight_stage_fake_context *fake = NULL;
    m3_stage_open_probe probe = {0};
    m3_backend_allocation_stats stats;
    m3_weight_stage_io io;
    m3_weight_stage stage;
    m3_weight_stage preserved;
    m3_backend *backend = NULL;
    m3_error error;
    bool success;

    m3_weight_stage_init(&stage);
    if (!m3_weight_stage_test_fixture_create(&fixture, &error) ||
        !m3_weight_stage_test_fake_backend_create(
            64U, 0U, SIZE_MAX, &backend, &fake, &error) ||
        m3_weight_stage_load(&stage, &fixture.table, backend, NULL, NULL,
                             &error) != M3_STATUS_OK) {
        m3_weight_stage_dispose(&stage);
        m3_backend_free(backend);
        (void)m3_weight_stage_test_fixture_dispose(&fixture);
        return false;
    }
    preserved = stage;
    fake->fail_allocation_call = fake->allocation_calls + 2U;
    probe.pass_through = true;
    m3_weight_stage_io_init(&io);
    io.context = &probe;
    io.open_file = m3_stage_probe_open;
    io.close_file = m3_stage_probe_close;
    success = m3_weight_stage_load_with_io(
                  &stage, &fixture.table, backend, NULL, NULL, &io, &error) ==
                  M3_STATUS_OUT_OF_MEMORY &&
              memcmp(&stage, &preserved, sizeof(stage)) == 0 &&
              probe.open_calls == 2U && probe.close_calls == 2U &&
              fake->allocation_calls == 4U && fake->free_calls == 1U &&
              m3_backend_get_allocation_stats(backend, &stats, &error) ==
                  M3_STATUS_OK &&
              stats.live_allocated_bytes == 22U &&
              stats.live_storage_count == 2U &&
              stats.peak_allocated_bytes == 32U &&
              stats.peak_storage_count == 3U;
    m3_weight_stage_dispose(&stage);
    success = success && fake->free_calls == 3U;
    m3_backend_free(backend);
    if (!m3_weight_stage_test_fixture_dispose(&fixture)) {
        success = false;
    }
    return success;
}

static bool m3_stage_working_set_overflow_case(void)
{
    const uint64_t payload_bytes = UINT64_MAX - UINT64_C(3);
    const uint64_t shape[] = {payload_bytes / UINT64_C(4)};
    m3_weight_stage_fake_context *fake = NULL;
    m3_stage_open_probe probe = {0};
    m3_weight_shard_record shard;
    m3_weight_binding binding;
    m3_weight_table table;
    m3_weight_stage_io io;
    m3_weight_stage stage;
    m3_backend *backend = NULL;
    m3_storage *existing = NULL;
    m3_error error;
    bool success;

    (void)memset(&shard, 0, sizeof(shard));
    (void)memset(&binding, 0, sizeof(binding));
    m3_weight_table_init(&table);
    m3_weight_stage_init(&stage);
    shard.path = "never-open.safetensors";
    shard.payload_bytes = payload_bytes;
    shard.snapshot.device = 1U;
    shard.snapshot.inode = 2U;
    shard.snapshot.file_size = payload_bytes;
    shard.snapshot.regular_file = true;
    binding.name = "huge";
    binding.shard_index = 0U;
    binding.data_end = payload_bytes;
    table.shards = &shard;
    table.shard_count = 1U;
    table.bindings = &binding;
    table.binding_count = 1U;
    table.aggregate_payload_bytes = payload_bytes;
    if (m3_tensor_metadata_init(&binding.tensor, M3_DTYPE_F32, 1U, shape,
                                &error) != M3_STATUS_OK ||
        !m3_weight_stage_test_fake_backend_create(
            UINT64_MAX, 0U, SIZE_MAX, &backend, &fake, &error) ||
        m3_storage_allocate(backend, 8U, 64U, &existing, &error) !=
            M3_STATUS_OK) {
        m3_storage_free(existing);
        m3_backend_free(backend);
        return false;
    }
    m3_weight_stage_io_init(&io);
    io.context = &probe;
    io.open_file = m3_stage_probe_open;
    success = m3_weight_stage_load_with_io(
                  &stage, &table, backend, NULL, NULL, &io, &error) ==
                  M3_STATUS_OVERFLOW &&
              probe.open_calls == 0U && fake->allocation_calls == 1U;
    m3_weight_stage_dispose(&stage);
    m3_storage_free(existing);
    m3_backend_free(backend);
    return success;
}

void m3_test_weight_stage_preflight_limits(m3_test_context *test)
{
    M3_TEST_EXPECT(test, m3_stage_storage_limit_case(9U),
                   "per-buffer limit fails before opening or allocating");
    M3_TEST_EXPECT(test, m3_stage_storage_limit_case(0U),
                   "zero backend allocation limit cannot stage weights");
    M3_TEST_EXPECT(test, m3_stage_working_set_case(),
                   "atomic replacement counts old and new working sets");
    M3_TEST_EXPECT(test, m3_stage_working_set_overflow_case(),
                   "live plus staged payload arithmetic is checked");
}

void m3_test_weight_stage_allocation_failure(m3_test_context *test)
{
    M3_TEST_EXPECT(test, m3_stage_allocation_failure_case(),
                   "partial backend allocation failure preserves old stage");
}

void m3_test_weight_stage_contract_rejections(m3_test_context *test)
{
    m3_weight_stage_test_fixture fixture;
    m3_weight_table empty;
    m3_weight_stage_io io;
    m3_weight_stage stage;
    m3_backend *backend = NULL;
    m3_error error;
    bool ready;

    m3_weight_table_init(&empty);
    m3_weight_stage_init(&stage);
    ready = m3_weight_stage_test_fixture_create(&fixture, &error) &&
            m3_backend_create_host(&backend, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "create staged-weight contract fixture");
    if (!ready) {
        m3_backend_free(backend);
        (void)m3_weight_stage_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_weight_stage_load(&stage, &empty, backend, NULL, NULL,
                                        NULL) == M3_STATUS_INVALID_FORMAT,
                   "empty component weight tables are rejected");
    m3_weight_stage_io_init(&io);
    io.maximum_chunk_bytes = 0U;
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_load_with_io(&stage, &fixture.table, backend, NULL,
                                     NULL, &io, &error) ==
            M3_STATUS_INVALID_ARGUMENT,
        "zero-sized internal read chunks are rejected");
    m3_weight_stage_io_init(&io);
    io.maximum_chunk_bytes = M3_WEIGHT_STAGE_MAXIMUM_CHUNK_BYTES + 1U;
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_load_with_io(&stage, &fixture.table, backend, NULL,
                                     NULL, &io, NULL) ==
            M3_STATUS_INVALID_ARGUMENT,
        "internal seams cannot exceed the production read bound");
    m3_weight_stage_io_init(&io);
    io.pread_file = NULL;
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_load_with_io(&stage, &fixture.table, backend, NULL,
                                     NULL, &io, &error) ==
            M3_STATUS_INVALID_ARGUMENT,
        "incomplete internal I/O seams are rejected");
    m3_weight_stage_dispose(&stage);
    m3_backend_free(backend);
    M3_TEST_EXPECT(test, m3_weight_stage_test_fixture_dispose(&fixture),
                   "remove staged-weight contract fixture");
}
