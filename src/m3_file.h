/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_FILE_H
#define M3_FILE_H

#include "m3_error.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t device;
    uint64_t inode;
    uint64_t file_size;
    int64_t modification_time_seconds;
    uint32_t modification_time_nanoseconds;
    int64_t change_time_seconds;
    uint32_t change_time_nanoseconds;
    bool regular_file;
} m3_file_snapshot;

m3_status m3_file_snapshot_descriptor(int descriptor,
                                       m3_file_snapshot *snapshot,
                                       m3_error *error);
bool m3_file_snapshot_equal(const m3_file_snapshot *left,
                            const m3_file_snapshot *right);
m3_status m3_file_read_bounded(const char *path, size_t maximum_size,
                               uint8_t **data, size_t *size,
                               m3_error *error);

#endif
