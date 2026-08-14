/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TOKENIZER_INTERNAL_H
#define M3_TOKENIZER_INTERNAL_H

#include "m3_tokenizer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define M3_TOKENIZER_JSON_MAX_BYTES (16U * 1024U * 1024U)
#define M3_TOKENIZER_SEMANTIC_FIRST 151675U
#define M3_TOKENIZER_SEMANTIC_LAST 168058U
#define M3_TOKENIZER_EMPTY_SLOT UINT32_MAX

typedef struct {
    uint8_t *data;
    size_t count;
    size_t capacity;
} m3_byte_buffer;

typedef struct {
    uint32_t offset;
    uint32_t length;
} m3_token_span;

typedef struct {
    uint32_t id;
    m3_token_span content;
    bool special;
} m3_added_token;

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t rank;
    uint32_t output;
} m3_merge_slot;

struct m3_tokenizer_state {
    uint32_t model_vocab_count;
    uint32_t merge_count;
    m3_token_span *model_tokens;
    uint8_t *model_bytes;
    size_t model_byte_count;
    uint32_t byte_ids[256];
    m3_added_token added[M3_TOKENIZER_ADDED_COUNT];
    uint8_t *added_bytes;
    size_t added_byte_count;
    m3_merge_slot *merge_slots;
    size_t merge_slot_count;
};

typedef struct {
    uint32_t model_vocab_count;
    uint32_t merge_count;
} m3_tokenizer_contract;

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t position;
    unsigned int depth;
} m3_tokenizer_json;

typedef struct {
    size_t start;
    size_t size;
    bool present;
} m3_json_span;

typedef struct {
    m3_json_span added_tokens;
    m3_json_span model;
} m3_tokenizer_sections;

typedef struct {
    m3_json_span vocab;
    m3_json_span merges;
} m3_model_sections;

m3_status m3_byte_buffer_reserve(m3_byte_buffer *buffer, size_t extra,
                                  m3_error *error);
m3_status m3_byte_buffer_append(m3_byte_buffer *buffer, const void *data,
                                 size_t size, m3_error *error);
void m3_byte_buffer_dispose(m3_byte_buffer *buffer);
bool m3_tokenizer_string_equal(const m3_byte_buffer *buffer,
                               m3_token_span span, const char *literal);

void m3_tokenizer_json_init(m3_tokenizer_json *json, const uint8_t *data,
                            size_t size);
bool m3_tokenizer_json_next_is(m3_tokenizer_json *json, uint8_t byte);
m3_status m3_tokenizer_json_expect(m3_tokenizer_json *json, uint8_t byte,
                                    m3_error *error);
m3_status m3_tokenizer_json_separator(m3_tokenizer_json *json,
                                       uint8_t closing, m3_error *error);
m3_status m3_tokenizer_json_string(m3_tokenizer_json *json,
                                    m3_byte_buffer *buffer,
                                    m3_token_span *span, m3_error *error);
m3_status m3_tokenizer_json_uint32(m3_tokenizer_json *json, uint32_t *value,
                                    m3_error *error);
m3_status m3_tokenizer_json_bool(m3_tokenizer_json *json, bool *value,
                                 m3_error *error);
m3_status m3_tokenizer_json_null(m3_tokenizer_json *json, m3_error *error);
m3_status m3_tokenizer_json_span(m3_tokenizer_json *json, m3_json_span *span,
                                 m3_error *error);
m3_status m3_tokenizer_json_finish(m3_tokenizer_json *json, m3_error *error);

m3_status m3_tokenizer_parse_sections(const uint8_t *data, size_t size,
                                       m3_tokenizer_sections *sections,
                                       m3_error *error);
m3_status m3_tokenizer_parse_model_sections(
    const uint8_t *data, const m3_json_span *span,
    const m3_tokenizer_contract *contract, m3_model_sections *sections,
    m3_error *error);
m3_status m3_tokenizer_state_build(const uint8_t *data, size_t size,
                                    const m3_tokenizer_contract *contract,
                                    m3_tokenizer_state **state,
                                    m3_error *error);
void m3_tokenizer_state_dispose(m3_tokenizer_state *state);
m3_status m3_tokenizer_load_data(m3_tokenizer *tokenizer,
                                  const uint8_t *data, size_t size,
                                  const m3_tokenizer_contract *contract,
                                  m3_error *error);

uint32_t m3_byte_alphabet_code_point(uint8_t byte);
bool m3_byte_alphabet_byte(uint32_t code_point, uint8_t *byte);
const m3_merge_slot *m3_tokenizer_find_merge(
    const m3_tokenizer_state *state, uint32_t left, uint32_t right);

#endif
