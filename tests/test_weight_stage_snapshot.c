/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "weight_stage_fixture.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    M3_STAGE_MUTATE_TRUNCATE = 0,
    M3_STAGE_MUTATE_REPLACE,
    M3_STAGE_MUTATE_TIMESTAMP,
    M3_STAGE_MUTATE_HARDLINK,
    M3_STAGE_MUTATE_FIFO
} m3_stage_mutation;

static bool m3_weight_stage_mutate_fixture(
    m3_weight_stage_test_fixture *fixture, m3_stage_mutation mutation)
{
    static const char a_header[] =
        "{\"gamma\":{\"dtype\":\"BF16\",\"shape\":[2],"
        "\"data_offsets\":[6,10]},"
        "\"beta\":{\"dtype\":\"F16\",\"shape\":[3],"
        "\"data_offsets\":[0,6]}}";
    char replacement[M3_TEST_PATH_CAPACITY];
    struct timespec times[2];

    switch (mutation) {
    case M3_STAGE_MUTATE_TRUNCATE:
        return truncate(fixture->a_path,
                        (off_t)(fixture->table.shards[0].snapshot.file_size -
                                1U)) == 0;
    case M3_STAGE_MUTATE_REPLACE:
        return m3_loader_test_path(replacement, fixture->root,
                                   "replacement.safetensors") &&
               m3_loader_test_write_safetensors(replacement, a_header, 10U) &&
               rename(replacement, fixture->a_path) == 0;
    case M3_STAGE_MUTATE_TIMESTAMP:
        times[0].tv_sec = 0;
        times[0].tv_nsec = UTIME_OMIT;
        times[1].tv_sec =
            (time_t)(fixture->table.shards[0]
                         .snapshot.modification_time_seconds +
                     1);
        times[1].tv_nsec = (long)fixture->table.shards[0]
                                .snapshot.modification_time_nanoseconds;
        return utimensat(AT_FDCWD, fixture->a_path, times, 0) == 0;
    case M3_STAGE_MUTATE_HARDLINK:
        return unlink(fixture->z_path) == 0 &&
               link(fixture->a_path, fixture->z_path) == 0;
    case M3_STAGE_MUTATE_FIFO:
        return unlink(fixture->z_path) == 0 &&
               mkfifo(fixture->z_path, 0600) == 0;
    }
    return false;
}

static bool m3_weight_stage_snapshot_rejection(m3_stage_mutation mutation,
                                                bool null_error)
{
    m3_weight_stage_test_fixture fixture;
    m3_backend_allocation_stats stats;
    m3_weight_stage stage;
    m3_weight_stage preserved;
    m3_backend *backend = NULL;
    m3_error error;
    bool success;

    m3_weight_stage_init(&stage);
    if (!m3_weight_stage_test_fixture_create(&fixture, &error) ||
        m3_backend_create_host(&backend, &error) != M3_STATUS_OK ||
        m3_weight_stage_load(&stage, &fixture.table, backend, NULL, NULL,
                             &error) != M3_STATUS_OK) {
        m3_weight_stage_dispose(&stage);
        m3_backend_free(backend);
        (void)m3_weight_stage_test_fixture_dispose(&fixture);
        return false;
    }
    preserved = stage;
    success = m3_weight_stage_mutate_fixture(&fixture, mutation) &&
              m3_weight_stage_load(&stage, &fixture.table, backend, NULL,
                                   NULL, null_error ? NULL : &error) ==
                  M3_STATUS_INVALID_FORMAT &&
              memcmp(&stage, &preserved, sizeof(stage)) == 0 &&
              m3_backend_get_allocation_stats(backend, &stats, &error) ==
                  M3_STATUS_OK &&
              stats.live_allocated_bytes == 22U &&
              stats.live_storage_count == 2U &&
              stats.peak_allocated_bytes == 22U &&
              stats.peak_storage_count == 2U;
    m3_weight_stage_dispose(&stage);
    m3_backend_free(backend);
    if (!m3_weight_stage_test_fixture_dispose(&fixture)) {
        success = false;
    }
    return success;
}

void m3_test_weight_stage_snapshot_rejections(m3_test_context *test)
{
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_snapshot_rejection(M3_STAGE_MUTATE_TRUNCATE, true),
        "truncated retained shard is rejected without allocations");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_snapshot_rejection(M3_STAGE_MUTATE_REPLACE, false),
        "same-size path replacement cannot replace the old stage");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_snapshot_rejection(M3_STAGE_MUTATE_TIMESTAMP, false),
        "same-inode metadata mutation is detected before payload reads");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_snapshot_rejection(M3_STAGE_MUTATE_HARDLINK, false),
        "hardlink path substitution cannot alias a retained shard");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_snapshot_rejection(M3_STAGE_MUTATE_FIFO, false),
        "nonblocking FIFO substitution is rejected as nonregular");
}
