/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_runtime_workspace.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool m3_weight_resolved_metadata_equal(
    const m3_tensor_metadata *left, const m3_tensor_metadata *right)
{
    uint8_t axis;

    if (left->dtype != right->dtype || left->rank != right->rank ||
        left->element_count != right->element_count ||
        left->byte_count != right->byte_count) {
        return false;
    }
    for (axis = 0U; axis < left->rank; ++axis) {
        if (left->shape[axis] != right->shape[axis]) {
            return false;
        }
    }
    return true;
}

m3_status m3_weight_stage_resolve_required(
    const m3_weight_stage *stage,
    const m3_weight_requirement *requirements, size_t requirement_count,
    const m3_tensor_view **views, m3_error *error)
{
    const m3_tensor_view **resolved = NULL;
    size_t index;
    m3_status status;

    if (stage == NULL || stage->table == NULL ||
        (requirement_count != 0U && views == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "weight stage and view outputs are required");
    }
    if (requirement_count > SIZE_MAX / sizeof(*resolved)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "resolved weight view array overflows");
    }
    status = m3_weight_table_validate_required(
        stage->table, requirements, requirement_count, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (requirement_count != 0U) {
        resolved = malloc(requirement_count * sizeof(*resolved));
        if (resolved == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate resolved weight views");
        }
    }
    for (index = 0U; index < requirement_count; ++index) {
        resolved[index] = m3_weight_stage_find_view(
            stage, requirements[index].name);
        if (resolved[index] == NULL) {
            free(resolved);
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "staged required weight '%s' is missing",
                                requirements[index].name);
        }
        if (!m3_weight_resolved_metadata_equal(
                &resolved[index]->metadata,
                &requirements[index].tensor)) {
            free(resolved);
            return m3_error_set(
                error, M3_STATUS_INVALID_FORMAT,
                "staged required weight '%s' has the wrong dtype or shape",
                requirements[index].name);
        }
    }
    if (requirement_count != 0U) {
        (void)memcpy(views, resolved,
                     requirement_count * sizeof(*resolved));
    }
    free(resolved);
    m3_error_reset(error);
    return M3_STATUS_OK;
}
