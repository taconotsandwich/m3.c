/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_PROMPT_INTERNAL_H
#define M3_PROMPT_INTERNAL_H

#include "m3_prompt.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} m3_text_builder;

bool m3_ascii_space(char byte);
bool m3_horizontal_space(char byte);
void m3_builder_dispose(m3_text_builder *builder);
m3_status m3_builder_append(m3_text_builder *builder, const char *data,
                            size_t length, m3_error *error);
m3_status m3_builder_byte(m3_text_builder *builder, char byte,
                          m3_error *error);
m3_status m3_builder_finish(m3_text_builder *builder, m3_prompt_text *text,
                            m3_error *error);
m3_status m3_normalize_input(m3_text_view input, const char *name,
                             m3_prompt_text *normalized, m3_error *error);
size_t m3_find_bytes(const char *data, size_t length, size_t start,
                     const char *needle, size_t needle_length);
m3_status m3_prompt_clean_caption(const m3_prompt_text *input,
                                  m3_prompt_text *output, m3_error *error);
m3_status m3_prompt_clean_lyrics(const m3_prompt_text *input,
                                 m3_prompt_text *output, m3_error *error);

#endif
