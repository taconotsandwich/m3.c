/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_VOCODER_RUNTIME_H
#define M3_VOCODER_RUNTIME_H

#include "m3_progress.h"
#include "m3_tensor.h"
#include "m3_weight_stage.h"

typedef struct m3_vocoder_runtime m3_vocoder_runtime;

typedef struct {
    m3_storage *storage;
    m3_tensor_view waveform;
} m3_vocoder_output;

void m3_vocoder_output_init(m3_vocoder_output *output);
void m3_vocoder_output_dispose(m3_vocoder_output *output);

/* Success owns all 91 materialized tensors and retains no stage/table views,
 * so the stage may be disposed immediately. The borrowed stage backend must
 * outlive the runtime. */
m3_status m3_vocoder_runtime_create(
    m3_vocoder_runtime **runtime, const m3_weight_stage *vocoder,
    m3_progress_callback progress, void *progress_context,
    m3_error *error);
void m3_vocoder_runtime_free(m3_vocoder_runtime *runtime);

/* The runtime and contiguous F32 latents [1,128,L] are borrowed. Official
 * L is 1..689. Success replaces output with an owned contiguous planar F32
 * waveform [1,2,512L]; failure or cancellation leaves output unchanged. */
m3_status m3_vocoder_decode_chunk(
    m3_vocoder_runtime *runtime, const m3_tensor_view *latents,
    m3_progress_callback progress, void *progress_context,
    m3_vocoder_output *output, m3_error *error);

#endif
