/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_JSON_H
#define M3_JSON_H

#include "m3_error.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_JSON_MAX_DEPTH 64U

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t position;
    unsigned int depth;
} m3_json_reader;

void m3_json_reader_init(m3_json_reader *reader, const uint8_t *data,
                         size_t size);
bool m3_json_next_is(m3_json_reader *reader, uint8_t byte);
m3_status m3_json_expect(m3_json_reader *reader, uint8_t byte,
                         m3_error *error);
m3_status m3_json_read_string(m3_json_reader *reader, char **value,
                              m3_error *error);
m3_status m3_json_read_uint64(m3_json_reader *reader, uint64_t *value,
                              m3_error *error);
m3_status m3_json_skip_value(m3_json_reader *reader, m3_error *error);
m3_status m3_json_finish(m3_json_reader *reader, m3_error *error);
m3_status m3_json_validate(const uint8_t *data, size_t size, m3_error *error);

#endif
