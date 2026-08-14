/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WAVEFORM_INTERNAL_H
#define M3_WAVEFORM_INTERNAL_H

#include "m3_vocoder_runtime.h"

typedef m3_status (*m3_waveform_read_fn)(
    void *context, const m3_storage *storage, size_t byte_offset,
    void *destination, size_t byte_count, m3_error *error);
typedef m3_status (*m3_waveform_write_fn)(
    void *context, m3_storage *storage, size_t byte_offset,
    const void *source, size_t byte_count, m3_error *error);

typedef struct {
    void *context;
    m3_waveform_read_fn read_storage;
    m3_waveform_write_fn write_storage;
} m3_waveform_io;

m3_status m3_waveform_assemble_io(
    const m3_tensor_view *chunks, size_t chunk_count,
    uint64_t frame_count, const m3_waveform_io *io,
    m3_progress_callback progress, void *progress_context,
    m3_vocoder_output *output, m3_error *error);

#endif
