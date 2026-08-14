/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_WEIGHT_STAGE_INTERNAL_H
#define M3_WEIGHT_STAGE_INTERNAL_H

#include "m3_weight_stage.h"

#include <sys/types.h>

#define M3_WEIGHT_STAGE_MAXIMUM_CHUNK_BYTES (16U * 1024U * 1024U)

typedef struct {
    void *context;
    int (*open_file)(void *context, const char *path, int flags);
    ssize_t (*pread_file)(void *context, int descriptor, void *buffer,
                          size_t byte_count, off_t byte_offset);
    int (*close_file)(void *context, int descriptor);
    m3_status (*write_storage)(void *context, m3_storage *storage,
                               size_t byte_offset, const void *source,
                               size_t byte_count, m3_error *error);
    size_t maximum_chunk_bytes;
} m3_weight_stage_io;

void m3_weight_stage_io_init(m3_weight_stage_io *io);
m3_status m3_weight_stage_load_with_io(
    m3_weight_stage *stage, const m3_weight_table *table,
    m3_backend *backend, m3_progress_callback progress,
    void *progress_context, const m3_weight_stage_io *io, m3_error *error);

m3_status m3_weight_stage_open_shards(const m3_weight_table *table,
                                       const m3_weight_stage_io *io,
                                       int *descriptors, m3_error *error);
m3_status m3_weight_stage_read_shards(
    m3_weight_stage *stage, const m3_weight_stage_io *io,
    const int *descriptors, m3_progress_callback progress,
    void *progress_context, m3_error *error);
m3_status m3_weight_stage_verify_shards(const m3_weight_table *table,
                                         const int *descriptors,
                                         m3_error *error);
m3_status m3_weight_stage_close_shards(const m3_weight_table *table,
                                        const m3_weight_stage_io *io,
                                        int *descriptors, bool report_errors,
                                        m3_error *error);

#endif
