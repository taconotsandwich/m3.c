/* SPDX-License-Identifier: GPL-2.0-only */

#include "vocoder_runtime_test.h"

#include "m3_test.h"

#include <string.h>

static void m3_vocoder_test_release_stage_only(
    m3_vocoder_test_fixture *fixture)
{
    m3_weight_stage_dispose(&fixture->stage);
    m3_weight_table_dispose(&fixture->table);
    m3_vocoder_plan_dispose(&fixture->plan);
}

void test_vocoder_stage_lifetime_independence(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    const m3_vocoder_weights *weights;
    m3_backend *backend = NULL;
    m3_backend_allocation_stats stats;
    m3_error error;
    float before[6];
    float after[6];
    size_t runtime_bytes;

    m3_error_reset(&error);
    M3_TEST_EXPECT(test,
                   m3_backend_create_host(&backend, &error) == M3_STATUS_OK &&
                       m3_vocoder_test_fixture_create(
                           &fixture, backend, false, &error),
                   "create stage-lifetime host fixture");
    if (fixture.backend == NULL) {
        m3_backend_free(backend);
        return;
    }
    runtime_bytes = m3_vocoder_test_runtime_bytes(&fixture);
    m3_vocoder_materialize_io_init(&io);
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_OK,
                   "materialize stage-independent runtime");
    if (runtime == NULL) {
        m3_vocoder_test_release_stage_only(&fixture);
        m3_backend_free(backend);
        return;
    }
    weights = m3_vocoder_runtime_weights(runtime);
    M3_TEST_EXPECT(test,
                   weights != NULL &&
                       m3_vocoder_test_read_values(
                           weights->decoder_input_weight, before, 6U,
                           &error),
                   "read owned tensor before disposing stage");
    if (weights == NULL) {
        m3_vocoder_runtime_free(runtime);
        m3_vocoder_test_release_stage_only(&fixture);
        m3_backend_free(backend);
        return;
    }
    m3_vocoder_test_release_stage_only(&fixture);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       backend, &stats, &error) == M3_STATUS_OK &&
                       stats.live_allocated_bytes == runtime_bytes &&
                       stats.live_storage_count == runtime->weights.count,
                   "disposing stage leaves only immutable runtime ownership");
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_read_values(
                       weights->decoder_input_weight, after, 6U, &error) &&
                       memcmp(before, after, sizeof(before)) == 0,
                   "owned tensor remains valid after immediate stage disposal");
    m3_vocoder_runtime_free(runtime);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       backend, &stats, &error) == M3_STATUS_OK &&
                       stats.live_allocated_bytes == 0U &&
                       stats.live_storage_count == 0U,
                   "runtime free releases all backend-owned weights");
    m3_backend_free(backend);
}

void test_vocoder_real_metal_ownership(m3_test_context *test)
{
    m3_vocoder_test_fixture fixture = {0};
    m3_vocoder_materialize_io io;
    m3_vocoder_runtime *runtime = NULL;
    m3_backend *backend = NULL;
    const m3_vocoder_weights *weights;
    m3_backend_allocation_stats stats;
    m3_error error;
    float copied[6];
    size_t index;
    m3_status status;

    m3_error_reset(&error);
    status = m3_backend_create_metal(&backend, &error);
    if (status == M3_STATUS_UNSUPPORTED) {
        M3_TEST_SKIP(test, m3_error_message(&error));
        return;
    }
    M3_TEST_EXPECT(test, status == M3_STATUS_OK,
                   "create real Metal backend for vocoder ownership");
    if (status != M3_STATUS_OK) {
        return;
    }
    M3_TEST_EXPECT(test,
                   m3_vocoder_test_fixture_create(
                       &fixture, backend, false, &error),
                   "create reduced Metal vocoder stage");
    if (fixture.backend == NULL) {
        m3_backend_free(backend);
        return;
    }
    m3_vocoder_materialize_io_init(&io);
    M3_TEST_EXPECT(test,
                   m3_vocoder_runtime_create_core(
                       &runtime, &fixture.stage, &fixture.plan, &io,
                       NULL, NULL, &error) == M3_STATUS_OK,
                   "materialize all reduced weights into Metal storage");
    if (runtime == NULL) {
        m3_vocoder_test_release_stage_only(&fixture);
        m3_backend_free(backend);
        return;
    }
    for (index = 0U; index < runtime->weights.count; ++index) {
        M3_TEST_EXPECT(test,
                       m3_storage_backend(
                           runtime->weights.views[index].storage) == backend,
                       "every materialized tensor is owned by Metal backend");
    }
    weights = m3_vocoder_runtime_weights(runtime);
    m3_vocoder_test_release_stage_only(&fixture);
    M3_TEST_EXPECT(test,
                   weights != NULL &&
                       m3_vocoder_test_read_values(
                           weights->decoder_input_weight, copied, 6U,
                           &error),
                   "Metal-owned tensor survives stage disposal");
    m3_vocoder_runtime_free(runtime);
    M3_TEST_EXPECT(test,
                   m3_backend_get_allocation_stats(
                       backend, &stats, &error) == M3_STATUS_OK &&
                       stats.live_allocated_bytes == 0U &&
                       stats.live_storage_count == 0U,
                   "Metal destination ownership releases completely");
    m3_backend_free(backend);
}
