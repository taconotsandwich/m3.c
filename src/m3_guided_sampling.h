/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_GUIDED_SAMPLING_H
#define M3_GUIDED_SAMPLING_H

#include "m3_backend.h"
#include "m3_tensor.h"

#include <stdbool.h>
#include <stdint.h>

#define M3_GUIDED_SEMANTIC_CODE_COUNT 16384U
#define M3_GUIDED_RESIDUAL_CODE_COUNT 1024U

typedef struct {
    bool eos;
    /* Code is meaningful only when eos is false. */
    uint32_t code;
} m3_guided_semantic_sample;

m3_status m3_guided_sample_semantic(
    const m3_backend *backend, const m3_tensor_view *eos_logits,
    const m3_tensor_view *semantic_logits, float uniform,
    m3_guided_semantic_sample *sample, m3_error *error);

m3_status m3_guided_sample_residual(
    const m3_backend *backend, const m3_tensor_view *logits, float uniform,
    uint32_t *code, m3_error *error);

#endif
