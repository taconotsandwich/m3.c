/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_VOCODER_RUNTIME_H
#define M3_VOCODER_RUNTIME_H

#include "m3_progress.h"
#include "m3_weight_stage.h"

typedef struct m3_vocoder_runtime m3_vocoder_runtime;

/* Success owns all 91 materialized tensors and retains no stage/table views,
 * so the stage may be disposed immediately. The borrowed stage backend must
 * outlive the runtime. */
m3_status m3_vocoder_runtime_create(
    m3_vocoder_runtime **runtime, const m3_weight_stage *vocoder,
    m3_progress_callback progress, void *progress_context,
    m3_error *error);
void m3_vocoder_runtime_free(m3_vocoder_runtime *runtime);

#endif
