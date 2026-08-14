/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TOKENIZER_MODEL_INTERNAL_H
#define M3_TOKENIZER_MODEL_INTERNAL_H

#include "m3_tokenizer_internal.h"

typedef struct {
    m3_token_span name;
    uint32_t id;
} m3_vocab_name;

typedef struct {
    m3_tokenizer_state *state;
    const m3_tokenizer_contract *contract;
    m3_vocab_name *names;
    uint32_t name_count;
    m3_byte_buffer name_bytes;
    uint32_t *name_slots;
    size_t name_slot_count;
} m3_tokenizer_build;

m3_status m3_tokenizer_build_init(m3_tokenizer_build *build,
                                   const m3_tokenizer_contract *contract,
                                   m3_error *error);
void m3_tokenizer_build_dispose(m3_tokenizer_build *build,
                                bool keep_state);
m3_status m3_tokenizer_parse_vocab(m3_tokenizer_build *build,
                                    const uint8_t *data,
                                    const m3_json_span *span,
                                    m3_error *error);
m3_status m3_tokenizer_parse_added(m3_tokenizer_build *build,
                                    const uint8_t *data,
                                    const m3_json_span *span,
                                    m3_error *error);
m3_status m3_tokenizer_parse_merges(m3_tokenizer_build *build,
                                     const uint8_t *data,
                                     const m3_json_span *span,
                                     m3_error *error);
bool m3_tokenizer_name_lookup(const m3_tokenizer_build *build,
                              const uint8_t *name, size_t length,
                              uint32_t *id);
m3_status m3_tokenizer_name_insert(m3_tokenizer_build *build,
                                    m3_token_span name, uint32_t id,
                                    m3_error *error);
size_t m3_tokenizer_hash_capacity(size_t count);
uint64_t m3_tokenizer_hash_bytes(const uint8_t *data, size_t size);
m3_status m3_tokenizer_merge_insert(m3_tokenizer_state *state,
                                     uint32_t left, uint32_t right,
                                     uint32_t rank, uint32_t output,
                                     m3_error *error);

#endif
