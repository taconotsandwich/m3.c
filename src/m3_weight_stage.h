/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WEIGHT_STAGE_H
#define M3_WEIGHT_STAGE_H

#include "m3_backend.h"
#include "m3_tensor.h"
#include "m3_weights.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef bool (*m3_weight_stage_progress)(void *context,
                                         uint64_t completed_bytes,
                                         uint64_t total_bytes);

/* Table is an unchanged builder/inspector result; table/backend are borrowed. */
/* Both inputs outlive the stage, and the table remains immutable until dispose. */
typedef struct {
    const m3_weight_table *table;
    m3_backend *backend;
    m3_storage **storages;
    size_t storage_count;
    m3_tensor_view *views;
    size_t view_count;
    uint64_t loaded_bytes;
} m3_weight_stage;

void m3_weight_stage_init(m3_weight_stage *stage);
void m3_weight_stage_dispose(m3_weight_stage *stage);
m3_status m3_weight_stage_load(m3_weight_stage *stage,
                               const m3_weight_table *table,
                               m3_backend *backend,
                               m3_weight_stage_progress progress,
                               void *progress_context, m3_error *error);
const m3_tensor_view *m3_weight_stage_find_view(
    const m3_weight_stage *stage, const char *name);

#endif
