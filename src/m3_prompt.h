/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_PROMPT_H
#define M3_PROMPT_H

#include "m3_error.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *data;
    size_t length;
} m3_text_view;

typedef struct {
    char *data;
    size_t length;
} m3_prompt_text;

typedef struct {
    const uint8_t *data;
    size_t length;
} m3_tokenizer_bytes;

void m3_prompt_text_init(m3_prompt_text *prompt);
void m3_prompt_text_dispose(m3_prompt_text *prompt);
m3_tokenizer_bytes m3_prompt_tokenizer_bytes(const m3_prompt_text *prompt);
m3_status m3_prompt_build(m3_prompt_text *prompt, m3_text_view caption,
                          m3_text_view lyrics, m3_error *error);

#endif
