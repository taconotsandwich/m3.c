/* SPDX-License-Identifier: GPL-2.0-only */

#ifndef M3_TOKENIZER_FIXTURE_H
#define M3_TOKENIZER_FIXTURE_H

#include "m3_tokenizer_internal.h"

typedef struct {
    uint32_t left;
    uint32_t right;
} m3_fixture_merge;

typedef enum {
    M3_TOKENIZER_FIXTURE_VALID = 0,
    M3_TOKENIZER_FIXTURE_DUPLICATE_VOCAB,
    M3_TOKENIZER_FIXTURE_DUPLICATE_MERGE,
    M3_TOKENIZER_FIXTURE_MISSING_MARKER,
    M3_TOKENIZER_FIXTURE_WRONG_ADDED_ID,
    M3_TOKENIZER_FIXTURE_EMPTY_PATTERN,
    M3_TOKENIZER_FIXTURE_EMPTY_MODEL_FIELD,
    M3_TOKENIZER_FIXTURE_EMPTY_VOCAB_TOKEN,
    M3_TOKENIZER_FIXTURE_EMPTY_MERGE_TOKEN
} m3_tokenizer_fixture_kind;

typedef struct {
    uint8_t *data;
    size_t size;
    m3_tokenizer_contract contract;
} m3_tokenizer_fixture;

bool m3_tokenizer_fixture_build(m3_tokenizer_fixture *fixture,
                                const m3_fixture_merge *merges,
                                size_t merge_count,
                                m3_tokenizer_fixture_kind kind);
void m3_tokenizer_fixture_dispose(m3_tokenizer_fixture *fixture);

#endif
