/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_runtime_workspace.h"
#include "weight_stage_fixture.h"

#include <stdint.h>
#include <string.h>

static void m3_runtime_spec(m3_runtime_tensor_spec *spec, m3_dtype dtype,
                            uint8_t rank, const uint64_t *shape,
                            size_t alignment)
{
    uint8_t index;

    (void)memset(spec, 0, sizeof(*spec));
    spec->dtype = dtype;
    spec->rank = rank;
    spec->alignment = alignment;
    for (index = 0U; index < rank && index < M3_TENSOR_MAX_RANK; ++index) {
        spec->shape[index] = shape[index];
    }
}

void m3_test_runtime_workspace_views(m3_test_context *test)
{
    const uint64_t matrix_shape[] = {2U, 3U};
    const uint64_t vector_shape[] = {8U};
    m3_runtime_tensor_spec specs[2];
    m3_backend_allocation_stats stats;
    m3_runtime_workspace workspace;
    m3_tensor_view *matrix;
    m3_tensor_view *vector;
    m3_backend *backend = NULL;
    m3_error error;
    bool ready;

    m3_runtime_workspace_init(NULL);
    m3_runtime_workspace_dispose(NULL);
    m3_runtime_workspace_init(&workspace);
    m3_runtime_spec(&specs[0], M3_DTYPE_F32, 2U, matrix_shape, 16U);
    m3_runtime_spec(&specs[1], M3_DTYPE_BF16, 1U, vector_shape, 64U);
    ready = m3_backend_create_host(&backend, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "create workspace host backend");
    if (!ready) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, specs, 2U, &error) ==
            M3_STATUS_OK,
        "build checked runtime workspace");
    matrix = m3_runtime_workspace_view(&workspace, 0U);
    vector = m3_runtime_workspace_view(&workspace, 1U);
    M3_TEST_EXPECT(
        test,
        workspace.backend == backend && workspace.count == 2U &&
            workspace.allocated_bytes == 40U && matrix != NULL &&
            vector != NULL && m3_runtime_workspace_view(&workspace, 2U) ==
                                  NULL &&
            m3_runtime_workspace_view(NULL, 0U) == NULL,
        "workspace provides deterministic indexed views");
    M3_TEST_EXPECT(
        test,
        matrix != NULL && matrix->metadata.dtype == M3_DTYPE_F32 &&
            matrix->metadata.rank == 2U &&
            matrix->metadata.shape[0] == 2U &&
            matrix->metadata.shape[1] == 3U &&
            matrix->metadata.byte_count == 24U &&
            matrix->byte_offset == 0U && matrix->byte_strides[0] == 12U &&
            matrix->byte_strides[1] == 4U &&
            m3_tensor_is_contiguous(matrix),
        "first workspace view has exact contiguous metadata");
    M3_TEST_EXPECT(
        test,
        matrix != NULL && vector != NULL &&
            vector->metadata.dtype == M3_DTYPE_BF16 &&
            vector->metadata.shape[0] == 8U &&
            vector->metadata.byte_count == 16U &&
            m3_storage_size(vector->storage) == 16U &&
            (uintptr_t)m3_storage_data(matrix->storage) % 16U == 0U &&
            (uintptr_t)m3_storage_data(vector->storage) % 64U == 0U,
        "workspace storages honor exact sizes and alignments");
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 40U &&
            stats.live_storage_count == 2U,
        "workspace allocations appear in backend statistics");
    m3_runtime_workspace_dispose(&workspace);
    M3_TEST_EXPECT(
        test,
        workspace.backend == NULL && workspace.storages == NULL &&
            workspace.views == NULL && workspace.count == 0U &&
            m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 0U &&
            stats.live_storage_count == 0U,
        "workspace disposal releases every owned storage");
    m3_backend_free(backend);
}

static void m3_test_runtime_backend_limits(m3_test_context *test)
{
    const uint64_t five_shape[] = {5U};
    const uint64_t four_shape[] = {4U};
    m3_weight_stage_fake_context *fake = NULL;
    m3_runtime_tensor_spec specs[2];
    m3_runtime_workspace workspace;
    m3_backend *backend = NULL;
    m3_error error;

    m3_runtime_workspace_init(&workspace);
    m3_runtime_spec(&specs[0], M3_DTYPE_F32, 1U, five_shape, 16U);
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_test_fake_backend_create(
            16U, 1024U, 0U, &backend, &fake, &error),
        "create per-buffer-limit backend");
    if (backend != NULL) {
        M3_TEST_EXPECT(
            test,
            m3_runtime_workspace_build(&workspace, backend, specs, 1U,
                                       &error) == M3_STATUS_OUT_OF_RANGE &&
                fake->allocation_calls == 0U && workspace.count == 0U,
            "workspace rejects a tensor above backend per-storage limit");
    }
    m3_backend_free(backend);
    backend = NULL;
    fake = NULL;
    m3_runtime_spec(&specs[0], M3_DTYPE_F32, 1U, four_shape, 16U);
    specs[1] = specs[0];
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_test_fake_backend_create(
            64U, 24U, 0U, &backend, &fake, &error),
        "create working-set-limit backend");
    if (backend != NULL) {
        M3_TEST_EXPECT(
            test,
            m3_runtime_workspace_build(&workspace, backend, specs, 2U,
                                       &error) == M3_STATUS_OUT_OF_MEMORY &&
                fake->allocation_calls == 0U && workspace.count == 0U,
            "workspace rejects aggregate above recommended working set");
    }
    m3_runtime_workspace_dispose(&workspace);
    m3_backend_free(backend);
}

void m3_test_runtime_workspace_failures(m3_test_context *test)
{
    const uint64_t one_shape[] = {1U};
    const uint64_t overflow_shape[] = {(uint64_t)SIZE_MAX};
    const uint64_t aggregate_shape[] = {
        (uint64_t)(SIZE_MAX / sizeof(float))};
    m3_runtime_tensor_spec aggregate_specs[2];
    m3_runtime_tensor_spec spec;
    m3_runtime_workspace workspace;
    m3_backend_allocation_stats stats;
    m3_backend *backend = NULL;
    m3_error error;
    bool ready;

    m3_test_runtime_backend_limits(test);
    m3_runtime_workspace_init(&workspace);
    ready = m3_backend_create_host(&backend, &error) == M3_STATUS_OK;
    M3_TEST_EXPECT(test, ready, "create workspace validation backend");
    if (!ready) {
        return;
    }
    m3_runtime_spec(&spec, M3_DTYPE_F32, 1U, one_shape, 24U);
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, &spec, 1U, &error) ==
            M3_STATUS_INVALID_ARGUMENT,
        "workspace rejects non-power-of-two alignment");
    spec.alignment = 2U;
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, &spec, 1U, NULL) ==
            M3_STATUS_INVALID_ARGUMENT,
        "workspace reports alignment failure without an error sink");
    m3_runtime_spec(&spec, M3_DTYPE_F32, 1U, overflow_shape, 16U);
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, &spec, 1U, &error) ==
            M3_STATUS_OVERFLOW,
        "workspace rejects overflowing tensor metadata");
    m3_runtime_spec(&aggregate_specs[0], M3_DTYPE_F32, 1U,
                    aggregate_shape, 16U);
    aggregate_specs[1] = aggregate_specs[0];
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, aggregate_specs, 2U,
                                   &error) == M3_STATUS_OVERFLOW,
        "workspace rejects overflowing aggregate tensor bytes");
    m3_runtime_spec(&spec, M3_DTYPE_F32, 1U, one_shape, 16U);
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, &spec, SIZE_MAX,
                                   &error) == M3_STATUS_OVERFLOW,
        "workspace rejects overflowing state array count before access");
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(NULL, backend, &spec, 1U, &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            m3_runtime_workspace_build(&workspace, NULL, &spec, 1U,
                                       &error) == M3_STATUS_INVALID_ARGUMENT &&
            m3_runtime_workspace_build(&workspace, backend, NULL, 1U,
                                       &error) == M3_STATUS_INVALID_ARGUMENT,
        "workspace rejects incomplete lifetime inputs");
    M3_TEST_EXPECT(
        test,
        m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 0U &&
            stats.live_storage_count == 0U,
        "workspace preflight failures allocate no storage");
    m3_runtime_workspace_dispose(&workspace);
    m3_backend_free(backend);
}

void m3_test_runtime_workspace_atomic(m3_test_context *test)
{
    const uint64_t two_shape[] = {2U};
    const uint64_t four_shape[] = {4U};
    m3_weight_stage_fake_context *fake = NULL;
    m3_runtime_tensor_spec old_spec;
    m3_runtime_tensor_spec replacement[2];
    m3_backend_allocation_stats stats;
    m3_runtime_workspace workspace;
    m3_tensor_view *preserved_view;
    m3_backend *backend = NULL;
    m3_error error;
    size_t free_calls;
    bool ready;

    m3_runtime_workspace_init(&workspace);
    m3_runtime_spec(&old_spec, M3_DTYPE_F32, 1U, two_shape, 16U);
    m3_runtime_spec(&replacement[0], M3_DTYPE_F32, 1U, four_shape, 16U);
    replacement[1] = replacement[0];
    ready = m3_weight_stage_test_fake_backend_create(
        1024U, 1024U, 0U, &backend, &fake, &error);
    M3_TEST_EXPECT(test, ready, "create transactional workspace backend");
    if (!ready) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, &old_spec, 1U,
                                   &error) == M3_STATUS_OK,
        "establish workspace state for atomic replacement");
    preserved_view = workspace.views;
    free_calls = fake->free_calls;
    fake->fail_allocation_call = fake->allocation_calls + 2U;
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, replacement, 2U,
                                   &error) == M3_STATUS_OUT_OF_MEMORY &&
            workspace.views == preserved_view && workspace.count == 1U &&
            workspace.allocated_bytes == 8U &&
            fake->free_calls == free_calls + 1U &&
            m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 8U &&
            stats.live_storage_count == 1U,
        "partial allocation failure cleans new state and preserves old state");
    fake->fail_allocation_call = 0U;
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, replacement, 2U,
                                   &error) == M3_STATUS_OK &&
            workspace.views != preserved_view && workspace.count == 2U &&
            workspace.allocated_bytes == 32U &&
            m3_backend_get_allocation_stats(backend, &stats, &error) ==
                M3_STATUS_OK &&
            stats.live_allocated_bytes == 32U &&
            stats.live_storage_count == 2U,
        "successful replacement publishes new state then releases old state");
    m3_runtime_workspace_dispose(&workspace);
    m3_backend_free(backend);

    backend = NULL;
    fake = NULL;
    m3_runtime_workspace_init(&workspace);
    ready = m3_weight_stage_test_fake_backend_create(
        1024U, 32U, 0U, &backend, &fake, &error);
    M3_TEST_EXPECT(test, ready, "create live-plus-new workspace backend");
    if (!ready) {
        return;
    }
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, &old_spec, 1U,
                                   &error) == M3_STATUS_OK,
        "establish state for live-plus-new preflight");
    preserved_view = workspace.views;
    M3_TEST_EXPECT(
        test,
        m3_runtime_workspace_build(&workspace, backend, replacement, 2U,
                                   &error) == M3_STATUS_OUT_OF_MEMORY &&
            fake->allocation_calls == 1U && workspace.views == preserved_view &&
            workspace.allocated_bytes == 8U,
        "replacement preflight counts current live plus complete new workspace");
    m3_runtime_workspace_dispose(&workspace);
    m3_backend_free(backend);
}
