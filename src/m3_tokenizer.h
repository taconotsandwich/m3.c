/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TOKENIZER_H
#define M3_TOKENIZER_H

#include "m3_prompt.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_TOKENIZER_MODEL_VOCAB_COUNT 151643U
#define M3_TOKENIZER_MERGE_COUNT 151387U
#define M3_TOKENIZER_ADDED_COUNT 32U
#define M3_TOKENIZER_ID_COUNT 151675U
#define M3_TOKENIZER_MAX_PROMPT_IDS 5000U

#define M3_TOKEN_END_OF_TEXT 151643U
#define M3_TOKEN_IM_START 151644U
#define M3_TOKEN_IM_END 151645U
#define M3_TOKEN_AUDIO_CFG 151654U
#define M3_TOKEN_AUDIO_START 151669U
#define M3_TOKEN_AUDIO_END 151670U
#define M3_TOKEN_CAPTION_START 151671U
#define M3_TOKEN_CAPTION_END 151672U
#define M3_TOKEN_LYRICS_START 151673U
#define M3_TOKEN_LYRICS_END 151674U

typedef struct m3_tokenizer_state m3_tokenizer_state;

typedef struct {
    m3_tokenizer_state *state;
} m3_tokenizer;

typedef struct {
    uint32_t *data;
    size_t count;
} m3_token_ids;

typedef struct {
    uint8_t *data;
    size_t length;
} m3_tokenizer_text;

void m3_tokenizer_init(m3_tokenizer *tokenizer);
void m3_tokenizer_dispose(m3_tokenizer *tokenizer);
m3_status m3_tokenizer_load(m3_tokenizer *tokenizer, const char *path,
                            m3_error *error);
void m3_token_ids_init(m3_token_ids *ids);
void m3_token_ids_dispose(m3_token_ids *ids);
void m3_tokenizer_text_init(m3_tokenizer_text *text);
void m3_tokenizer_text_dispose(m3_tokenizer_text *text);
m3_status m3_tokenizer_encode(const m3_tokenizer *tokenizer,
                              m3_tokenizer_bytes input, m3_token_ids *ids,
                              m3_error *error);
m3_status m3_tokenizer_decode(const m3_tokenizer *tokenizer,
                              const uint32_t *ids, size_t count,
                              bool skip_special, m3_tokenizer_text *text,
                              m3_error *error);
m3_status m3_tokenizer_build_cfg_row(const uint32_t *prompt_ids,
                                     size_t prompt_count,
                                     m3_token_ids *cfg_ids,
                                     m3_error *error);

#endif
