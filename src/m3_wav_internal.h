/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WAV_INTERNAL_H
#define M3_WAV_INTERNAL_H

#include "m3_wav.h"

#include <stddef.h>
#include <sys/types.h>

typedef struct {
    void *context;
    int (*create_temporary)(void *context, char *path_template);
    ssize_t (*write_bytes)(void *context, int descriptor,
                           const void *data, size_t size);
    int (*sync_file)(void *context, int descriptor);
    int (*close_file)(void *context, int descriptor);
    int (*replace_file)(void *context, const char *source,
                        const char *destination);
    int (*remove_file)(void *context, const char *path);
} m3_wav_io;

const m3_wav_io *m3_wav_default_io(void);

m3_status m3_wav_write_f32_with_io(const char *path,
                                   const m3_tensor_view *samples,
                                   const m3_wav_io *io,
                                   m3_error *error);

#endif
