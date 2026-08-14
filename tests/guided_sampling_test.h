/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_GUIDED_SAMPLING_TEST_H
#define M3_GUIDED_SAMPLING_TEST_H

#include "m3_op_test.h"

#include <stdbool.h>
#include <stddef.h>

bool m3_guided_test_tensor(m3_op_test_fixture *fixture,
                           m3_tensor_view *view, m3_dtype dtype,
                           size_t candidate_count, float initial);
void m3_guided_test_set(m3_tensor_view *view, size_t row,
                        size_t candidate, float value);

#endif
