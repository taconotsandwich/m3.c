/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_FLOW_RUNTIME_TEST_H
#define M3_FLOW_RUNTIME_TEST_H

#include "m3_flow_internal.h"
#include "m3_op_test.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    m3_op_test_fixture fixture;
    m3_flow_config config;
    m3_flow_weights weights;
    m3_condition_weights condition_weights;
    m3_tensor_view weight_views[25];
    size_t weight_count;
    m3_tensor_view frames;
} m3_flow_test_fixture;

bool m3_flow_test_fixture_init(m3_flow_test_fixture *fixture,
                               uint64_t frame_count);
bool m3_flow_test_fixture_init_music3(
    m3_flow_test_fixture *fixture, uint64_t frame_count,
    m3_backend *backend);
void m3_flow_test_fixture_dispose(m3_flow_test_fixture *fixture);
bool m3_flow_test_strided_frames(m3_flow_test_fixture *fixture,
                                 uint64_t frame_count,
                                 m3_tensor_view *frames);
bool m3_flow_test_run_init(m3_flow_test_fixture *fixture,
                           uint64_t maximum_length, m3_flow_run *run);
void m3_flow_test_run_dispose(m3_flow_run *run);

#endif
