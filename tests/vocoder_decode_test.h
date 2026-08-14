/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_VOCODER_DECODE_TEST_H
#define M3_VOCODER_DECODE_TEST_H

#include "m3_vocoder_internal.h"

#include <stdbool.h>

void m3_vocoder_decode_test_config(m3_vocoder_plan_config *config);
bool m3_vocoder_decode_test_runtime(
    m3_backend *backend, m3_vocoder_runtime **runtime, m3_error *error);
bool m3_vocoder_decode_test_runtime_config(
    m3_backend *backend, const m3_vocoder_plan_config *config,
    m3_vocoder_runtime **runtime, m3_error *error);
bool m3_vocoder_decode_test_latents(
    m3_backend *backend, uint64_t length, const float *values,
    m3_storage **storage, m3_tensor_view *view, m3_error *error);

#endif
