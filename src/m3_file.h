/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_FILE_H
#define M3_FILE_H

#include "m3_error.h"

#include <stddef.h>
#include <stdint.h>

m3_status m3_file_read_bounded(const char *path, size_t maximum_size,
                               uint8_t **data, size_t *size,
                               m3_error *error);

#endif
