/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_SEMANTIC_RUNTIME_H
#define M3_SEMANTIC_RUNTIME_H

#include "m3_progress.h"
#include "m3_rng.h"
#include "m3_tensor.h"
#include "m3_weight_stage.h"

#include <stdint.h>

#define M3_SEMANTIC_MAX_FRAMES 9000U

typedef struct {
    m3_storage *storage;
    /* Owned contiguous BF16 [1,F,8,4096] semantic conditioning. */
    m3_tensor_view frame_hiddens;
} m3_semantic_output;

void m3_semantic_output_init(m3_semantic_output *output);
void m3_semantic_output_dispose(m3_semantic_output *output);

/* Both immutable stages and their common backend are borrowed for the call.
 * Prompt IDs are contiguous I32 [2,T] conditional then unconditional rows.
 * On failure or cancellation, output and rng remain unchanged. */
m3_status m3_semantic_generate(
    const m3_weight_stage *language_model,
    const m3_weight_stage *rvq_depth_decoder,
    const m3_tensor_view *prompt_ids, uint64_t frame_limit, m3_rng *rng,
    m3_progress_callback progress, void *progress_context,
    m3_semantic_output *output, m3_error *error);

#endif
