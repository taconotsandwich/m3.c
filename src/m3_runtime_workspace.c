/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_runtime_workspace.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool m3_runtime_power_of_two(size_t value)
{
    return value != 0U && (value & (value - 1U)) == 0U;
}

void m3_runtime_workspace_init(m3_runtime_workspace *workspace)
{
    if (workspace != NULL) {
        (void)memset(workspace, 0, sizeof(*workspace));
    }
}

void m3_runtime_workspace_dispose(m3_runtime_workspace *workspace)
{
    size_t index;

    if (workspace == NULL) {
        return;
    }
    if (workspace->storages != NULL) {
        for (index = 0U; index < workspace->count; ++index) {
            size_t later;

            for (later = index + 1U; later < workspace->count;
                 ++later) {
                if (workspace->storages[later] ==
                    workspace->storages[index]) {
                    workspace->storages[later] = NULL;
                }
            }
            m3_storage_free(workspace->storages[index]);
        }
    }
    free(workspace->storages);
    free(workspace->views);
    m3_runtime_workspace_init(workspace);
}

static m3_status m3_runtime_workspace_arrays(
    m3_runtime_workspace *built, m3_tensor_metadata **metadata,
    m3_error *error)
{
    if (built->count == 0U) {
        return M3_STATUS_OK;
    }
    built->storages = calloc(built->count, sizeof(*built->storages));
    built->views = calloc(built->count, sizeof(*built->views));
    *metadata = calloc(built->count, sizeof(**metadata));
    if (built->storages == NULL || built->views == NULL ||
        *metadata == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate runtime workspace state");
    }
    return M3_STATUS_OK;
}

static m3_status m3_runtime_workspace_metadata(
    const m3_runtime_tensor_spec *specs, size_t spec_count,
    const m3_backend_info *info, m3_tensor_metadata *metadata,
    size_t *aggregate, m3_error *error)
{
    size_t index;

    *aggregate = 0U;
    for (index = 0U; index < spec_count; ++index) {
        m3_status status;

        if (!m3_runtime_power_of_two(specs[index].alignment) ||
            specs[index].alignment < sizeof(void *)) {
            return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                                "runtime tensor alignment is invalid");
        }
        status = m3_tensor_metadata_init(
            &metadata[index], specs[index].dtype, specs[index].rank,
            specs[index].shape, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if ((uint64_t)metadata[index].byte_count >
            info->maximum_storage_bytes) {
            return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                "runtime tensor exceeds backend storage limit");
        }
        if (*aggregate > SIZE_MAX - metadata[index].byte_count) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "runtime workspace byte total overflows");
        }
        *aggregate += metadata[index].byte_count;
    }
    return M3_STATUS_OK;
}

static m3_status m3_runtime_workspace_working_set(
    m3_backend *backend, const m3_backend_info *info, size_t aggregate,
    m3_error *error)
{
    m3_backend_allocation_stats stats;
    uint64_t live;
    uint64_t added;
    m3_status status = m3_backend_get_allocation_stats(backend, &stats,
                                                       error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    live = (uint64_t)stats.live_allocated_bytes;
    added = (uint64_t)aggregate;
    if ((size_t)live != stats.live_allocated_bytes ||
        (size_t)added != aggregate || added > UINT64_MAX - live) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "runtime workspace working set overflows");
    }
    if (info->recommended_working_set_bytes != 0U &&
        live + added > info->recommended_working_set_bytes) {
        return m3_error_set(
            error, M3_STATUS_OUT_OF_MEMORY,
            "runtime workspace exceeds backend recommended working set");
    }
    return M3_STATUS_OK;
}

static m3_status m3_runtime_workspace_allocate(
    m3_runtime_workspace *built, const m3_runtime_tensor_spec *specs,
    const m3_tensor_metadata *metadata, m3_error *error)
{
    size_t index;

    for (index = 0U; index < built->count; ++index) {
        m3_status status = m3_storage_allocate(
            built->backend, metadata[index].byte_count,
            specs[index].alignment, &built->storages[index], error);

        if (status != M3_STATUS_OK) {
            return status;
        }
        status = m3_tensor_view_contiguous(
            &built->views[index], built->storages[index],
            metadata[index].dtype, metadata[index].rank,
            metadata[index].shape, 0U, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_runtime_workspace_build(
    m3_runtime_workspace *workspace, m3_backend *backend,
    const m3_runtime_tensor_spec *specs, size_t spec_count,
    m3_error *error)
{
    m3_runtime_workspace built;
    m3_tensor_metadata *metadata = NULL;
    m3_backend_info info;
    size_t aggregate = 0U;
    m3_status status;

    if (workspace == NULL || backend == NULL ||
        (spec_count != 0U && specs == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "workspace, backend, and tensor specs are required");
    }
    if (spec_count > SIZE_MAX / sizeof(*built.storages) ||
        spec_count > SIZE_MAX / sizeof(*built.views) ||
        spec_count > SIZE_MAX / sizeof(*metadata)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "runtime workspace state allocation overflows");
    }
    m3_runtime_workspace_init(&built);
    built.backend = backend;
    built.count = spec_count;
    status = m3_backend_get_info(backend, &info, error);
    if (status == M3_STATUS_OK) {
        status = m3_runtime_workspace_arrays(&built, &metadata, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_runtime_workspace_metadata(
            specs, spec_count, &info, metadata, &aggregate, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_runtime_workspace_working_set(
            backend, &info, aggregate, error);
    }
    if (status == M3_STATUS_OK) {
        built.allocated_bytes = aggregate;
        status = m3_runtime_workspace_allocate(
            &built, specs, metadata, error);
    }
    free(metadata);
    if (status != M3_STATUS_OK) {
        m3_runtime_workspace_dispose(&built);
        return status;
    }
    m3_runtime_workspace_dispose(workspace);
    *workspace = built;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_tensor_view *m3_runtime_workspace_view(m3_runtime_workspace *workspace,
                                          size_t index)
{
    if (workspace == NULL || index >= workspace->count ||
        workspace->views == NULL) {
        return NULL;
    }
    return &workspace->views[index];
}
