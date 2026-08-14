/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_scratch.h"

#include <stdbool.h>
#include <stdint.h>

static bool m3_scratch_power_of_two(size_t value)
{
    return value != 0U && (value & (value - 1U)) == 0U;
}

m3_status m3_scratch_arena_init(m3_scratch_arena *arena, void *backing,
                                size_t capacity, m3_error *error)
{
    if (arena == NULL || (capacity != 0U && backing == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "scratch arena and nonempty backing are required");
    }
    arena->data = backing;
    arena->capacity = capacity;
    arena->offset = 0U;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

size_t m3_scratch_arena_mark(const m3_scratch_arena *arena)
{
    return arena == NULL ? 0U : arena->offset;
}

m3_status m3_scratch_arena_allocate(m3_scratch_arena *arena,
                                    size_t byte_count, size_t alignment,
                                    void **memory, m3_error *error)
{
    uintptr_t base;
    uintptr_t current;
    size_t aligned_offset;
    size_t end;
    size_t remainder;
    size_t padding;

    if (memory == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "scratch allocation output is null");
    }
    *memory = NULL;
    if (arena == NULL || arena->offset > arena->capacity) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "scratch arena state is invalid");
    }
    if (!m3_scratch_power_of_two(alignment)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "scratch alignment must be a power of two");
    }
    base = (uintptr_t)arena->data;
    if (arena->offset > (size_t)(UINTPTR_MAX - base)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "scratch address overflows uintptr_t");
    }
    current = base + (uintptr_t)arena->offset;
    remainder = (size_t)(current & (uintptr_t)(alignment - 1U));
    padding = remainder == 0U ? 0U : alignment - remainder;
    if (arena->offset > SIZE_MAX - padding) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "scratch aligned offset overflows");
    }
    aligned_offset = arena->offset + padding;
    if (aligned_offset > SIZE_MAX - byte_count) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "scratch allocation end overflows");
    }
    end = aligned_offset + byte_count;
    if (end > arena->capacity) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "scratch arena is exhausted");
    }
    if (arena->data != NULL) {
        *memory = arena->data + aligned_offset;
    }
    arena->offset = end;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_scratch_arena_rewind(m3_scratch_arena *arena, size_t mark,
                                  m3_error *error)
{
    if (arena == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "scratch arena is null");
    }
    if (mark > arena->offset) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "scratch rewind mark is ahead of current offset");
    }
    arena->offset = mark;
    m3_error_reset(error);
    return M3_STATUS_OK;
}

void m3_scratch_arena_reset(m3_scratch_arena *arena)
{
    if (arena != NULL) {
        arena->offset = 0U;
    }
}
