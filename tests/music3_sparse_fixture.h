/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MUSIC3_SPARSE_FIXTURE_H
#define M3_MUSIC3_SPARSE_FIXTURE_H

#include <stdbool.h>
#include <stdint.h>

bool m3_test_music3_create_sparse_layout(char root[256],
                                         uint64_t *logical_bytes,
                                         uint64_t *physical_bytes);

#endif
