/* SPDX-License-Identifier: GPL-2.0-only */

#include "guided_sampling_test.h"

#include "m3_op_internal.h"

#include <stdint.h>
#include <stdlib.h>

bool m3_guided_test_tensor(m3_op_test_fixture *fixture,
                           m3_tensor_view *view, m3_dtype dtype,
                           size_t candidate_count, float initial)
{
    const uint64_t shape[] = {2U, (uint64_t)candidate_count};
    m3_tensor_metadata metadata;
    m3_error error;
    void *values;
    size_t index;
    bool result;

    if (m3_tensor_metadata_init(&metadata, dtype, 2U, shape, &error) !=
        M3_STATUS_OK) {
        return false;
    }
    values = calloc(1U, metadata.byte_count);
    if (values == NULL) {
        return false;
    }
    result = m3_op_test_tensor(fixture, view, dtype, 2U, shape, values);
    free(values);
    if (!result) {
        return false;
    }
    for (index = 0U; index < metadata.element_count; ++index) {
        m3_op_store_float(view, m3_op_element_offset(view, index), initial);
    }
    return true;
}

void m3_guided_test_set(m3_tensor_view *view, size_t row,
                        size_t candidate, float value)
{
    size_t flat = row * (size_t)view->metadata.shape[1] + candidate;

    m3_op_store_float(view, m3_op_element_offset(view, flat), value);
}
