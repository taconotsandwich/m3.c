/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_METAL_DENSE_TEST_H
#define M3_METAL_DENSE_TEST_H

#include "m3_op_test.h"
#include "m3_test.h"

typedef struct {
    m3_op_test_fixture host;
    m3_op_test_fixture metal;
} m3_metal_dense_fixture;

typedef struct {
    m3_tensor_view host;
    m3_tensor_view metal;
} m3_metal_dense_view;

bool m3_metal_dense_fixture_init(m3_test_context *test,
                                 m3_metal_dense_fixture *fixture);
void m3_metal_dense_fixture_dispose(m3_metal_dense_fixture *fixture);
bool m3_metal_dense_tensor(m3_metal_dense_fixture *fixture,
                           m3_metal_dense_view *view, m3_dtype dtype,
                           uint8_t rank, const uint64_t *shape,
                           const void *values);
bool m3_metal_dense_strided(m3_metal_dense_fixture *fixture,
                            m3_metal_dense_view *view, m3_dtype dtype,
                            uint8_t rank, const uint64_t *shape,
                            const size_t *strides, size_t byte_count,
                            const void *values);
bool m3_metal_dense_execute(m3_metal_dense_fixture *fixture,
                            const m3_command *host_commands,
                            const m3_command *metal_commands,
                            size_t command_count, m3_error *error);
bool m3_metal_dense_equal(const m3_metal_dense_view *view);

#endif
