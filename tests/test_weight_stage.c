/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "weight_stage_fixture.h"

#include <string.h>

typedef struct {
    uint64_t completed[8];
    uint64_t total[8];
    size_t call_count;
} m3_weight_stage_progress_log;

static bool m3_weight_stage_record_progress(void *context,
                                            uint64_t completed,
                                            uint64_t total)
{
    m3_weight_stage_progress_log *log = context;

    if (log->call_count < sizeof(log->completed) / sizeof(log->completed[0])) {
        log->completed[log->call_count] = completed;
        log->total[log->call_count] = total;
    }
    ++log->call_count;
    return true;
}

static bool m3_weight_stage_view_bytes(const m3_weight_stage *stage,
                                       const char *name,
                                       const uint8_t *expected,
                                       size_t expected_size,
                                       m3_error *error)
{
    const m3_tensor_view *view = m3_weight_stage_find_view(stage, name);
    uint8_t actual[16] = {0U};

    return view != NULL && view->metadata.byte_count == expected_size &&
           m3_tensor_is_contiguous(view) && expected_size <= sizeof(actual) &&
           m3_storage_read(view->storage, view->byte_offset, actual,
                           expected_size, error) == M3_STATUS_OK &&
           memcmp(actual, expected, expected_size) == 0;
}

void m3_test_weight_stage_load(m3_test_context *test)
{
    static const uint8_t alpha[] = {1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
    static const uint8_t beta[] = {1U, 2U, 3U, 4U, 5U, 6U};
    static const uint8_t gamma[] = {7U, 8U, 9U, 10U};
    static const uint8_t zeta[] = {9U, 10U, 11U, 12U};
    m3_weight_stage_test_fixture fixture;
    m3_weight_stage_progress_log progress = {0};
    m3_backend_allocation_stats stats;
    m3_weight_stage stage;
    m3_backend *backend = NULL;
    m3_error error;
    bool ready = m3_weight_stage_test_fixture_create(&fixture, &error) &&
                 m3_backend_create_host(&backend, &error) == M3_STATUS_OK;

    m3_weight_stage_init(&stage);
    M3_TEST_EXPECT(test, ready, "create tiny staged-weight fixture and backend");
    if (!ready) {
        m3_backend_free(backend);
        (void)m3_weight_stage_test_fixture_dispose(&fixture);
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_load(&stage, &fixture.table, backend,
                             m3_weight_stage_record_progress, &progress,
                             &error) == M3_STATUS_OK,
        "load unsorted shard provenance into a staged component");
    M3_TEST_EXPECT(
        test,
        stage.table == &fixture.table && stage.backend == backend &&
            stage.storage_count == 2U && stage.view_count == 4U &&
            stage.loaded_bytes == 22U && progress.call_count == 3U &&
            progress.completed[0] == 0U && progress.completed[1] == 10U &&
            progress.completed[2] == 22U && progress.total[0] == 22U &&
            progress.total[1] == 22U && progress.total[2] == 22U,
        "publish exact storage, view, byte, and progress counts");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_view_bytes(&stage, "alpha", alpha, sizeof(alpha),
                                   &error) &&
            m3_weight_stage_view_bytes(&stage, "beta", beta, sizeof(beta),
                                       &error) &&
            m3_weight_stage_view_bytes(&stage, "gamma", gamma,
                                       sizeof(gamma), &error) &&
            m3_weight_stage_view_bytes(&stage, "zeta", zeta, sizeof(zeta),
                                       &error),
        "views preserve exact payload bytes, shapes, and binding offsets");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_find_view(&stage, "alpha")->storage ==
                stage.storages[1] &&
            m3_weight_stage_find_view(&stage, "beta")->storage ==
                stage.storages[0] &&
            m3_weight_stage_find_view(&stage, "missing") == NULL &&
            m3_weight_stage_find_view(&stage, NULL) == NULL,
        "find maps sorted binding names without copying names or paths");
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 22U &&
            stats.live_storage_count == 2U &&
            stats.peak_allocated_bytes == 22U &&
            stats.peak_storage_count == 2U,
        "backend statistics include one exact allocation per shard");
    {
        m3_weight_table malformed = fixture.table;
        m3_weight_stage preserved = stage;

        malformed.aggregate_payload_bytes = 23U;
        M3_TEST_EXPECT(
            test,
            m3_weight_stage_load(&stage, &malformed, backend, NULL, NULL,
                                 NULL) == M3_STATUS_INVALID_FORMAT &&
                memcmp(&stage, &preserved, sizeof(stage)) == 0,
            "failed replacement returns exact status and preserves old stage");
    }
    m3_weight_stage_dispose(&stage);
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 0U &&
            stats.live_storage_count == 0U &&
            stats.peak_allocated_bytes == 22U &&
            stats.peak_storage_count == 2U,
        "disposing a stage releases storage while retaining peak statistics");
    m3_backend_free(backend);
    M3_TEST_EXPECT(test, m3_weight_stage_test_fixture_dispose(&fixture),
                   "remove staged-weight fixture files");
}
