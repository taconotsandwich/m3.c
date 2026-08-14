/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_OP_INTERNAL_H
#define M3_OP_INTERNAL_H

#include "m3_op.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    float value;
    int32_t index;
} m3_top_pair;

bool m3_op_dtype_float(m3_dtype dtype);
bool m3_op_shape_equal(const m3_tensor_view *left,
                       const m3_tensor_view *right);
m3_status m3_op_check_view(const m3_backend *backend,
                           const m3_tensor_view *view, bool output,
                           const char *name, m3_error *error);
m3_status m3_op_check_alias(const m3_tensor_view *output,
                            const m3_tensor_view *input,
                            bool exact_allowed, const char *name,
                            m3_error *error);
m3_status m3_op_check_pair_disjoint(const m3_tensor_view *left,
                                    const m3_tensor_view *right,
                                    const char *name, m3_error *error);
m3_status m3_op_broadcast_validate(const m3_tensor_view *left,
                                   const m3_tensor_view *right,
                                   const m3_tensor_view *output,
                                   m3_error *error);
m3_status m3_op_validate_basic(const m3_backend *backend,
                               const m3_command *command,
                               bool *handled, m3_error *error);
m3_status m3_op_validate_nn(const m3_backend *backend,
                            const m3_command *command,
                            bool *handled, m3_error *error);
m3_status m3_op_validate_sampling(const m3_backend *backend,
                                  const m3_command *command,
                                  bool *handled, m3_error *error);
m3_status m3_op_command_scratch(const m3_command *command,
                                size_t *byte_count, m3_error *error);

size_t m3_op_element_offset(const m3_tensor_view *view, size_t flat_index);
size_t m3_op_broadcast_offset(const m3_tensor_view *input,
                              const m3_tensor_view *output,
                              size_t output_flat_index);
float m3_op_load_float(const m3_tensor_view *view, size_t byte_offset);
int32_t m3_op_load_i32(const m3_tensor_view *view, size_t byte_offset);
void m3_op_store_float(m3_tensor_view *view, size_t byte_offset,
                       float value);
void m3_op_store_i32(m3_tensor_view *view, size_t byte_offset,
                     int32_t value);

m3_status m3_host_execute_basic(const m3_command *command,
                                m3_scratch_arena *scratch,
                                bool *handled, m3_error *error);
m3_status m3_host_execute_dense(const m3_command *command,
                                m3_scratch_arena *scratch,
                                bool *handled, m3_error *error);
m3_status m3_host_execute_attention(const m3_command *command,
                                    m3_scratch_arena *scratch,
                                    bool *handled, m3_error *error);
m3_status m3_host_execute_sampling(const m3_command *command,
                                   m3_scratch_arena *scratch,
                                   bool *handled, m3_error *error);
m3_status m3_host_execute_commands(void *context,
                                   const m3_command *commands,
                                   size_t command_count,
                                   m3_scratch_arena *scratch,
                                   m3_error *error);

#endif
