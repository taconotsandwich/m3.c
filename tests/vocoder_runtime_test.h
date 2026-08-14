/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_VOCODER_RUNTIME_TEST_H
#define M3_VOCODER_RUNTIME_TEST_H

#include "m3_vocoder_internal.h"

#include <stdbool.h>

typedef struct {
    m3_vocoder_plan plan;
    m3_weight_table table;
    m3_weight_stage stage;
    m3_backend *backend;
    bool owns_backend;
} m3_vocoder_test_fixture;

void m3_vocoder_test_config(m3_vocoder_plan_config *config);
bool m3_vocoder_test_fixture_create(
    m3_vocoder_test_fixture *fixture, m3_backend *backend,
    bool owns_backend, m3_error *error);
void m3_vocoder_test_fixture_dispose(m3_vocoder_test_fixture *fixture);
m3_tensor_view *m3_vocoder_test_source(
    m3_vocoder_test_fixture *fixture, const char *name);
bool m3_vocoder_test_fill(m3_tensor_view *view, float value,
                          m3_error *error);
bool m3_vocoder_test_write_values(m3_tensor_view *view,
                                  const float *values, size_t count,
                                  m3_error *error);
bool m3_vocoder_test_read_values(const m3_tensor_view *view,
                                 float *values, size_t count,
                                 m3_error *error);
size_t m3_vocoder_test_source_bytes(const m3_vocoder_test_fixture *fixture);
size_t m3_vocoder_test_runtime_bytes(const m3_vocoder_test_fixture *fixture);

#endif
