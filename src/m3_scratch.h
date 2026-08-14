/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_SCRATCH_H
#define M3_SCRATCH_H

#include "m3_error.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *data;
    size_t capacity;
    size_t offset;
} m3_scratch_arena;

m3_status m3_scratch_arena_init(m3_scratch_arena *arena, void *backing,
                                size_t capacity, m3_error *error);
size_t m3_scratch_arena_mark(const m3_scratch_arena *arena);
m3_status m3_scratch_arena_allocate(m3_scratch_arena *arena,
                                    size_t byte_count, size_t alignment,
                                    void **memory, m3_error *error);
m3_status m3_scratch_arena_rewind(m3_scratch_arena *arena, size_t mark,
                                  m3_error *error);
void m3_scratch_arena_reset(m3_scratch_arena *arena);

#endif
