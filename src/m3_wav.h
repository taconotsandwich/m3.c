/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WAV_H
#define M3_WAV_H

#include "m3_error.h"

#include <stdint.h>

m3_status m3_wav_write_f32(const char *path, const float *samples,
                           uint32_t sample_rate, uint16_t channel_count,
                           uint64_t frame_count, m3_error *error);

#endif
