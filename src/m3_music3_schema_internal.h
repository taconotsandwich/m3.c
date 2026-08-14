/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_MUSIC3_SCHEMA_INTERNAL_H
#define M3_MUSIC3_SCHEMA_INTERNAL_H

#include "m3_music3_schema.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    m3_music3_schema_visitor visitor;
    void *context;
    m3_error *error;
} m3_music3_schema_emitter;

m3_status m3_music3_schema_emit(m3_music3_schema_emitter *emitter,
                                const char *name, m3_dtype dtype,
                                uint8_t rank, const uint64_t *shape);
m3_status m3_music3_schema_emit1(m3_music3_schema_emitter *emitter,
                                 const char *name, uint64_t first);
m3_status m3_music3_schema_emit2(m3_music3_schema_emitter *emitter,
                                 const char *name, uint64_t first,
                                 uint64_t second);
m3_status m3_music3_schema_emit3(m3_music3_schema_emitter *emitter,
                                 const char *name, uint64_t first,
                                 uint64_t second, uint64_t third);
m3_status m3_music3_schema_format(char *name, size_t capacity,
                                  m3_error *error, const char *format, ...);

m3_status m3_music3_schema_generate_lm(m3_music3_schema_emitter *emitter);
m3_status m3_music3_schema_generate_rvq(m3_music3_schema_emitter *emitter);
m3_status m3_music3_schema_generate_condition(
    m3_music3_schema_emitter *emitter);
m3_status m3_music3_schema_generate_flow(m3_music3_schema_emitter *emitter);
m3_status m3_music3_schema_generate_vocoder(m3_music3_schema_emitter *emitter);

#endif
