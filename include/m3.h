/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_H
#define M3_H

#define M3_VERSION_MAJOR 0
#define M3_VERSION_MINOR 1
#define M3_VERSION_PATCH 0
#define M3_VERSION_STRING "0.1.0"

typedef struct m3_ctx m3_ctx;

typedef enum {
    M3_STATUS_OK = 0,
    M3_STATUS_INVALID_ARGUMENT = 1,
    M3_STATUS_OUT_OF_RANGE = 2,
    M3_STATUS_OVERFLOW = 3,
    M3_STATUS_OUT_OF_MEMORY = 4,
    M3_STATUS_IO = 5,
    M3_STATUS_INVALID_FORMAT = 6,
    M3_STATUS_UNSUPPORTED = 7,
    M3_STATUS_CANCELLED = 8,
    M3_STATUS_INTERNAL = 9
} m3_status;

const char *m3_version(void);
const char *m3_status_string(m3_status status);

#endif
