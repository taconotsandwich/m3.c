/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_RUNTIME_WORKSPACE_H
#define M3_RUNTIME_WORKSPACE_H

#include "m3_backend.h"
#include "m3_op.h"
#include "m3_tensor.h"
#include "m3_weight_stage.h"
#include "m3_weights.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    m3_dtype dtype;
    uint8_t rank;
    uint64_t shape[M3_TENSOR_MAX_RANK];
    size_t alignment;
} m3_runtime_tensor_spec;

/* The backend is borrowed and must outlive the workspace. */
typedef struct {
    m3_backend *backend;
    m3_storage **storages;
    m3_tensor_view *views;
    size_t count;
    size_t allocated_bytes;
} m3_runtime_workspace;

void m3_runtime_workspace_init(m3_runtime_workspace *workspace);
void m3_runtime_workspace_dispose(m3_runtime_workspace *workspace);
m3_status m3_runtime_workspace_build(
    m3_runtime_workspace *workspace, m3_backend *backend,
    const m3_runtime_tensor_spec *specs, size_t spec_count,
    m3_error *error);
m3_tensor_view *m3_runtime_workspace_view(m3_runtime_workspace *workspace,
                                          size_t index);

/* The backend is borrowed and must outlive the executor. */
typedef struct {
    m3_backend *backend;
    void *scratch;
    size_t scratch_capacity;
} m3_command_executor;

void m3_command_executor_init(m3_command_executor *executor,
                              m3_backend *backend);
void m3_command_executor_dispose(m3_command_executor *executor);
m3_status m3_command_executor_execute(m3_command_executor *executor,
                                      const m3_command *commands,
                                      size_t command_count,
                                      m3_error *error);

m3_status m3_weight_stage_resolve_required(
    const m3_weight_stage *stage,
    const m3_weight_requirement *requirements, size_t requirement_count,
    const m3_tensor_view **views, m3_error *error);

#endif
