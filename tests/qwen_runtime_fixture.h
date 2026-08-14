/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_QWEN_RUNTIME_FIXTURE_H
#define M3_QWEN_RUNTIME_FIXTURE_H

#include "m3_op_test.h"
#include "m3_qwen_internal.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    m3_op_test_fixture tensors;
    m3_qwen_dimensions dimensions;
    m3_tensor_view weight_views[14];
    m3_qwen_weights weights;
    m3_qwen_cache cache;
    m3_runtime_workspace rope;
    m3_command_executor executor;
    m3_qwen_forward_state forward;
} m3_qwen_test_fixture;

bool m3_qwen_test_fixture_init(m3_qwen_test_fixture *fixture,
                               uint64_t capacity);
void m3_qwen_test_fixture_dispose(m3_qwen_test_fixture *fixture);
bool m3_qwen_test_ids(m3_qwen_test_fixture *fixture,
                      m3_tensor_view *view, uint64_t sequence,
                      const int32_t *values);
uint16_t m3_qwen_test_bf16_at(const m3_tensor_view *view, size_t index);
float m3_qwen_test_f32_at(const m3_tensor_view *view, size_t index);

#endif
