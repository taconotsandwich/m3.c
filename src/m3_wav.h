/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WAV_H
#define M3_WAV_H

#include "m3_error.h"
#include "m3_tensor.h"

m3_status m3_wav_write_f32(const char *path,
                           const m3_tensor_view *samples,
                           m3_error *error);

#endif
