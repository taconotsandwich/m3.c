/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_PROGRESS_H
#define M3_PROGRESS_H

#include <stdbool.h>
#include <stdint.h>

typedef bool (*m3_progress_callback)(void *context, uint64_t completed,
                                     uint64_t total);

#endif
