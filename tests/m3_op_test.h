/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_OP_TEST_H
#define M3_OP_TEST_H

#include "m3_backend.h"
#include "m3_op.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_OP_TEST_STORAGE_CAPACITY 48U

typedef struct {
    m3_backend *backend;
    m3_storage *storages[M3_OP_TEST_STORAGE_CAPACITY];
    size_t storage_count;
    bool owns_backend;
} m3_op_test_fixture;

bool m3_op_test_fixture_init(m3_op_test_fixture *fixture);
bool m3_op_test_fixture_init_backend(m3_op_test_fixture *fixture,
                                     m3_backend *backend,
                                     bool owns_backend);
void m3_op_test_fixture_dispose(m3_op_test_fixture *fixture);
bool m3_op_test_tensor(m3_op_test_fixture *fixture, m3_tensor_view *view,
                       m3_dtype dtype, uint8_t rank,
                       const uint64_t *shape, const void *values);
bool m3_op_test_storage(m3_op_test_fixture *fixture, size_t byte_count,
                        m3_storage **storage);
m3_status m3_op_test_execute(m3_op_test_fixture *fixture,
                             const m3_command *commands,
                             size_t command_count, size_t *scratch_bytes,
                             m3_error *error);
float *m3_op_test_f32(m3_tensor_view *view);
int32_t *m3_op_test_i32(m3_tensor_view *view);
uint16_t *m3_op_test_u16(m3_tensor_view *view);

#endif
