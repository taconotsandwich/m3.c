/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_runtime_workspace.h"

#include <string.h>

static bool m3_runtime_resolver_fixture(
    m3_backend **backend, m3_storage **alpha_storage,
    m3_storage **beta_storage, m3_tensor_view views[2],
    m3_weight_binding bindings[2], m3_weight_table *table,
    m3_weight_stage *stage, m3_error *error)
{
    const uint64_t alpha_shape[] = {2U};
    const uint64_t beta_shape[] = {3U};

    (void)memset(bindings, 0, 2U * sizeof(*bindings));
    (void)memset(table, 0, sizeof(*table));
    (void)memset(stage, 0, sizeof(*stage));
    m3_tensor_view_init(&views[0]);
    m3_tensor_view_init(&views[1]);
    if (m3_backend_create_host(backend, error) != M3_STATUS_OK ||
        m3_storage_allocate(*backend, 8U, 16U, alpha_storage, error) !=
            M3_STATUS_OK ||
        m3_storage_allocate(*backend, 6U, 16U, beta_storage, error) !=
            M3_STATUS_OK ||
        m3_tensor_view_contiguous(&views[0], *alpha_storage, M3_DTYPE_F32,
                                  1U, alpha_shape, 0U, error) !=
            M3_STATUS_OK ||
        m3_tensor_view_contiguous(&views[1], *beta_storage, M3_DTYPE_F16,
                                  1U, beta_shape, 0U, error) !=
            M3_STATUS_OK) {
        return false;
    }
    bindings[0].name = "alpha";
    bindings[0].tensor = views[0].metadata;
    bindings[1].name = "beta";
    bindings[1].tensor = views[1].metadata;
    table->bindings = bindings;
    table->binding_count = 2U;
    stage->table = table;
    stage->backend = *backend;
    stage->views = views;
    stage->view_count = 2U;
    return true;
}

static void m3_runtime_resolver_dispose(m3_backend *backend,
                                        m3_storage *alpha_storage,
                                        m3_storage *beta_storage)
{
    m3_storage_free(beta_storage);
    m3_storage_free(alpha_storage);
    m3_backend_free(backend);
}

static bool m3_runtime_requirement_metadata(
    m3_weight_requirement *requirement, const char *name, m3_dtype dtype,
    uint8_t rank, const uint64_t *shape, m3_error *error)
{
    requirement->name = name;
    return m3_tensor_metadata_init(&requirement->tensor, dtype, rank, shape,
                                   error) == M3_STATUS_OK;
}

void m3_test_runtime_weight_resolver(m3_test_context *test)
{
    const uint64_t alpha_shape[] = {2U};
    const uint64_t wrong_shape[] = {4U};
    const uint64_t wrong_rank_shape[] = {1U, 3U};
    m3_weight_binding bindings[2];
    m3_weight_requirement requirements[2];
    m3_weight_requirement wrong;
    m3_tensor_view stage_views[2];
    const m3_tensor_view *resolved[2];
    const m3_tensor_view *sentinel[2];
    m3_weight_table table;
    m3_weight_stage stage;
    m3_storage *alpha_storage = NULL;
    m3_storage *beta_storage = NULL;
    m3_backend *backend = NULL;
    m3_error error;
    bool ready;

    ready = m3_runtime_resolver_fixture(
        &backend, &alpha_storage, &beta_storage, stage_views, bindings,
        &table, &stage, &error);
    M3_TEST_EXPECT(test, ready, "create exact staged-view resolver fixture");
    if (!ready) {
        m3_runtime_resolver_dispose(backend, alpha_storage, beta_storage);
        return;
    }
    requirements[0].name = "beta";
    requirements[0].tensor = bindings[1].tensor;
    requirements[1].name = "alpha";
    requirements[1].tensor = bindings[0].tensor;
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(
            &stage, requirements, 2U, resolved, &error) == M3_STATUS_OK &&
            resolved[0] == &stage_views[1] &&
            resolved[1] == &stage_views[0],
        "resolver publishes exact views in declared requirement order");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, NULL, 0U, NULL, &error) ==
            M3_STATUS_OK,
        "resolver accepts an empty required array");
    sentinel[0] = &stage_views[0];
    sentinel[1] = &stage_views[1];
    resolved[0] = sentinel[0];
    resolved[1] = sentinel[1];
    ready = m3_runtime_requirement_metadata(
        &wrong, "missing", M3_DTYPE_F32, 1U, alpha_shape, &error);
    M3_TEST_EXPECT(test, ready, "create missing resolver requirement");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, &wrong, 1U, resolved,
                                         &error) == M3_STATUS_INVALID_FORMAT &&
            resolved[0] == sentinel[0] && resolved[1] == sentinel[1],
        "missing weight failure preserves all caller outputs");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, &wrong, 1U, resolved,
                                         NULL) == M3_STATUS_INVALID_FORMAT &&
            resolved[0] == sentinel[0],
        "missing weight returns exact status without an error sink");
    ready = m3_runtime_requirement_metadata(
        &wrong, "alpha", M3_DTYPE_BF16, 1U, alpha_shape, &error);
    M3_TEST_EXPECT(test, ready, "create wrong-dtype resolver requirement");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, &wrong, 1U, resolved,
                                         &error) == M3_STATUS_INVALID_FORMAT &&
            resolved[0] == sentinel[0],
        "resolver rejects a valid but wrong weight dtype atomically");
    ready = m3_runtime_requirement_metadata(
        &wrong, "beta", M3_DTYPE_F16, 2U, wrong_rank_shape, &error);
    M3_TEST_EXPECT(test, ready, "create wrong-rank resolver requirement");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, &wrong, 1U, resolved,
                                         &error) == M3_STATUS_INVALID_FORMAT &&
            resolved[0] == sentinel[0],
        "resolver rejects wrong rank without publishing output");
    ready = m3_runtime_requirement_metadata(
        &wrong, "beta", M3_DTYPE_F16, 1U, wrong_shape, &error);
    M3_TEST_EXPECT(test, ready, "create wrong-shape resolver requirement");
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, &wrong, 1U, resolved,
                                         &error) == M3_STATUS_INVALID_FORMAT &&
            resolved[0] == sentinel[0],
        "resolver rejects wrong shape without publishing output");
    wrong.tensor = bindings[1].tensor;
    stage_views[1].metadata.shape[0] = 4U;
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, &wrong, 1U, resolved,
                                         &error) == M3_STATUS_INVALID_FORMAT &&
            resolved[0] == sentinel[0],
        "resolver rejects staged metadata drift without publishing output");
    stage_views[1].metadata = bindings[1].tensor;
    wrong.name = NULL;
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, &wrong, 1U, resolved,
                                         &error) == M3_STATUS_INVALID_ARGUMENT &&
            resolved[0] == sentinel[0],
        "resolver reuses required-name validation atomically");
    wrong.name = "beta";
    wrong.tensor = bindings[1].tensor;
    stage.view_count = 1U;
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(&stage, &wrong, 1U, resolved,
                                         &error) == M3_STATUS_INVALID_FORMAT &&
            resolved[0] == sentinel[0],
        "resolver rejects an incomplete staged view inventory atomically");
    stage.view_count = 2U;
    M3_TEST_EXPECT(
        test,
        m3_weight_stage_resolve_required(NULL, &wrong, 1U, resolved,
                                         &error) == M3_STATUS_INVALID_ARGUMENT &&
            m3_weight_stage_resolve_required(&stage, &wrong, 1U, NULL,
                                             &error) ==
                M3_STATUS_INVALID_ARGUMENT &&
            resolved[0] == sentinel[0],
        "resolver rejects incomplete arguments without touching outputs");
    m3_runtime_resolver_dispose(backend, alpha_storage, beta_storage);
}
