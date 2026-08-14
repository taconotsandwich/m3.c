/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_RVQ_CONDITION_TEST_H
#define M3_RVQ_CONDITION_TEST_H

#include "m3_op_test.h"
#include "m3_rvq_condition_internal.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    m3_op_test_fixture fixture;
    m3_rvq_config config;
    m3_rvq_weights weights;
    m3_tensor_view last_hidden;
    m3_tensor_view semantic_embedding;
} m3_rvq_test_fixture;

typedef struct {
    m3_op_test_fixture fixture;
    m3_condition_config config;
    m3_condition_weights weights;
    m3_tensor_view weight_views[4];
    m3_tensor_view frames;
} m3_condition_test_fixture;

typedef struct {
    size_t call_count;
    uint64_t completed[16];
    uint64_t total[16];
    uint64_t cancel_at;
} m3_runtime_progress_log;

bool m3_rc_tensor(m3_op_test_fixture *fixture, m3_tensor_view *view,
                  m3_dtype dtype, uint8_t rank, const uint64_t *shape,
                  float initial);
void m3_rc_set(m3_tensor_view *view, size_t flat_index, float value);
float m3_rc_get(const m3_tensor_view *view, size_t flat_index);
bool m3_rc_progress(void *context, uint64_t completed, uint64_t total);

bool m3_rvq_test_fixture_init(m3_rvq_test_fixture *fixture,
                              m3_dtype dtype);
void m3_rvq_test_fixture_dispose(m3_rvq_test_fixture *fixture);
void m3_rvq_test_uniforms(float uniforms[M3_RVQ_RESIDUAL_COUNT],
                          uint32_t codes[M3_RVQ_RESIDUAL_COUNT]);

bool m3_condition_test_fixture_init(m3_condition_test_fixture *fixture);
void m3_condition_test_fixture_dispose(
    m3_condition_test_fixture *fixture);
float m3_condition_test_mixed(const m3_tensor_view *frames,
                              size_t frame_index);

#endif
