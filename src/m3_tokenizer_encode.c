/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_internal.h"

#include "m3_unicode.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t *data;
    size_t count;
    size_t capacity;
} m3_id_buffer;

static m3_status m3_ids_append(m3_id_buffer *ids, uint32_t id,
                                m3_error *error)
{
    if (ids->count >= M3_TOKENIZER_MAX_PROMPT_IDS) {
        return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                            "tokenized prompt exceeds %u IDs",
                            M3_TOKENIZER_MAX_PROMPT_IDS);
    }
    if (ids->count == ids->capacity) {
        size_t capacity = ids->capacity == 0U ? 64U : ids->capacity * 2U;
        uint32_t *data;

        if (capacity > M3_TOKENIZER_MAX_PROMPT_IDS) {
            capacity = M3_TOKENIZER_MAX_PROMPT_IDS;
        }
        if (capacity <= ids->capacity ||
            capacity > SIZE_MAX / sizeof(*ids->data)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "token ID buffer size overflows");
        }
        data = realloc(ids->data, capacity * sizeof(*ids->data));
        if (data == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate prompt token IDs");
        }
        ids->data = data;
        ids->capacity = capacity;
    }
    ids->data[ids->count++] = id;
    return M3_STATUS_OK;
}

static bool m3_peek(const uint8_t *data, size_t size, size_t position,
                    uint32_t *cp, size_t *length)
{
    return position < size &&
           m3_utf8_decode(data + position, size - position, cp, length);
}

static bool m3_is_newline(uint32_t cp)
{
    return cp == (uint32_t)'\r' || cp == (uint32_t)'\n';
}

static bool m3_is_symbol(uint32_t cp)
{
    return !m3_unicode_is_whitespace(cp) && !m3_unicode_is_letter(cp) &&
           !m3_unicode_is_number(cp);
}

static size_t m3_contraction_candidate(const uint8_t *data, size_t size,
                                        const char *suffix)
{
    size_t position = 1U;
    size_t index;

    for (index = 0U; suffix[index] != '\0'; ++index) {
        uint32_t cp;
        size_t length;

        if (!m3_peek(data, size, position, &cp, &length) ||
            m3_unicode_simple_fold(cp) != (uint32_t)(uint8_t)suffix[index]) {
            return 0U;
        }
        position += length;
    }
    return position;
}

static size_t m3_contraction(const uint8_t *data, size_t size)
{
    static const char *const suffixes[] = {
        "s", "t", "re", "ve", "m", "ll", "d",
    };
    size_t index;

    if (size == 0U || data[0] != (uint8_t)'\'') {
        return 0U;
    }
    for (index = 0U; index < sizeof(suffixes) / sizeof(suffixes[0]); ++index) {
        size_t length = m3_contraction_candidate(data, size, suffixes[index]);

        if (length != 0U) {
            return length;
        }
    }
    return 0U;
}

static size_t m3_letter_piece(const uint8_t *data, size_t size)
{
    uint32_t cp;
    size_t length;
    size_t position = 0U;

    if (!m3_peek(data, size, 0U, &cp, &length)) {
        return 0U;
    }
    if (m3_unicode_is_letter(cp)) {
        position = length;
    } else {
        uint32_t next;
        size_t next_length;

        if (m3_is_newline(cp) || m3_unicode_is_number(cp) ||
            !m3_peek(data, size, length, &next, &next_length) ||
            !m3_unicode_is_letter(next)) {
            return 0U;
        }
        position = length + next_length;
    }
    while (position < size) {
        size_t next_length;

        if (!m3_peek(data, size, position, &cp, &next_length) ||
            !m3_unicode_is_letter(cp)) {
            break;
        }
        position += next_length;
    }
    return position;
}

static size_t m3_number_piece(const uint8_t *data, size_t size)
{
    uint32_t cp;
    size_t length;

    return m3_peek(data, size, 0U, &cp, &length) &&
                   m3_unicode_is_number(cp)
               ? length
               : 0U;
}

static size_t m3_symbol_piece(const uint8_t *data, size_t size)
{
    uint32_t cp;
    size_t length;
    size_t position = 0U;

    if (!m3_peek(data, size, 0U, &cp, &length)) {
        return 0U;
    }
    if (cp == (uint32_t)' ') {
        uint32_t next;
        size_t next_length;

        if (!m3_peek(data, size, length, &next, &next_length) ||
            !m3_is_symbol(next)) {
            return 0U;
        }
        position = length + next_length;
    } else if (m3_is_symbol(cp)) {
        position = length;
    } else {
        return 0U;
    }
    while (position < size) {
        size_t next_length;

        if (!m3_peek(data, size, position, &cp, &next_length) ||
            !m3_is_symbol(cp)) {
            break;
        }
        position += next_length;
    }
    while (position < size) {
        size_t next_length;

        if (!m3_peek(data, size, position, &cp, &next_length) ||
            !m3_is_newline(cp)) {
            break;
        }
        position += next_length;
    }
    return position;
}

static size_t m3_newline_piece(const uint8_t *data, size_t size)
{
    uint32_t cp;
    size_t length;
    size_t position = 0U;
    size_t last_newline = 0U;

    while (position < size && m3_peek(data, size, position, &cp, &length) &&
           m3_unicode_is_whitespace(cp)) {
        position += length;
        if (m3_is_newline(cp)) {
            last_newline = position;
        }
    }
    return last_newline;
}

static size_t m3_whitespace_piece(const uint8_t *data, size_t size)
{
    uint32_t cp;
    size_t length;
    size_t position = 0U;
    size_t last_start = 0U;
    size_t scalar_count = 0U;

    while (position < size && m3_peek(data, size, position, &cp, &length) &&
           m3_unicode_is_whitespace(cp)) {
        last_start = position;
        position += length;
        scalar_count += 1U;
    }
    if (position < size && scalar_count >= 2U) {
        return last_start;
    }
    return position;
}

static size_t m3_next_piece(const uint8_t *data, size_t size)
{
    size_t length;

    length = m3_contraction(data, size);
    if (length == 0U) length = m3_letter_piece(data, size);
    if (length == 0U) length = m3_number_piece(data, size);
    if (length == 0U) length = m3_symbol_piece(data, size);
    if (length == 0U) length = m3_newline_piece(data, size);
    if (length == 0U) length = m3_whitespace_piece(data, size);
    return length;
}

static m3_status m3_bpe_piece(const m3_tokenizer_state *state,
                               const uint8_t *data, size_t size,
                               m3_id_buffer *output, m3_error *error)
{
    uint32_t *symbols;
    size_t symbol_count = size;
    size_t index;

    if (size == 0U) {
        return M3_STATUS_OK;
    }
    if (size > SIZE_MAX / sizeof(*symbols)) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "BPE piece size overflows");
    }
    symbols = malloc(size * sizeof(*symbols));
    if (symbols == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate BPE symbols");
    }
    for (index = 0U; index < size; ++index) {
        symbols[index] = state->byte_ids[data[index]];
    }
    while (symbol_count > 1U) {
        const m3_merge_slot *best = NULL;
        size_t best_position = 0U;

        for (index = 0U; index + 1U < symbol_count; ++index) {
            const m3_merge_slot *candidate = m3_tokenizer_find_merge(
                state, symbols[index], symbols[index + 1U]);

            if (candidate != NULL &&
                (best == NULL || candidate->rank < best->rank)) {
                best = candidate;
                best_position = index;
            }
        }
        if (best == NULL) {
            break;
        }
        symbols[best_position] = best->output;
        if (best_position + 2U < symbol_count) {
            (void)memmove(symbols + best_position + 1U,
                          symbols + best_position + 2U,
                          (symbol_count - best_position - 2U) *
                              sizeof(*symbols));
        }
        symbol_count -= 1U;
    }
    for (index = 0U; index < symbol_count; ++index) {
        m3_status status = m3_ids_append(output, symbols[index], error);

        if (status != M3_STATUS_OK) {
            free(symbols);
            return status;
        }
    }
    free(symbols);
    return M3_STATUS_OK;
}

static m3_status m3_encode_ordinary(const m3_tokenizer_state *state,
                                     const uint8_t *data, size_t size,
                                     m3_id_buffer *output, m3_error *error)
{
    uint8_t *normalized = NULL;
    size_t normalized_size = 0U;
    size_t position = 0U;
    m3_status status;

    if (size == 0U) {
        return M3_STATUS_OK;
    }
    status = m3_unicode_nfc(data, size, &normalized, &normalized_size, error);
    while (status == M3_STATUS_OK && position < normalized_size) {
        size_t piece_size = m3_next_piece(normalized + position,
                                          normalized_size - position);

        if (piece_size == 0U) {
            status = m3_error_set(error, M3_STATUS_INTERNAL,
                                  "pre-tokenizer scanner made no progress");
            break;
        }
        status = m3_bpe_piece(state, normalized + position, piece_size,
                              output, error);
        position += piece_size;
    }
    free(normalized);
    return status;
}

static const m3_added_token *m3_added_match(
    const m3_tokenizer_state *state, const uint8_t *data, size_t size)
{
    const m3_added_token *best = NULL;
    size_t index;

    for (index = 0U; index < M3_TOKENIZER_ADDED_COUNT; ++index) {
        const m3_added_token *candidate = &state->added[index];
        size_t length = candidate->content.length;

        if (length <= size && (best == NULL ||
                              length > (size_t)best->content.length) &&
            memcmp(data, state->added_bytes + candidate->content.offset,
                   length) == 0) {
            best = candidate;
        }
    }
    return best;
}

static m3_status m3_validate_utf8(const uint8_t *data, size_t size,
                                  m3_error *error)
{
    size_t position = 0U;

    while (position < size) {
        uint32_t cp;
        size_t length;

        if (!m3_utf8_decode(data + position, size - position, &cp, &length)) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "invalid tokenizer input UTF-8 at byte %zu",
                                position);
        }
        position += length;
    }
    return M3_STATUS_OK;
}

m3_status m3_tokenizer_encode(const m3_tokenizer *tokenizer,
                              m3_tokenizer_bytes input, m3_token_ids *ids,
                              m3_error *error)
{
    m3_id_buffer built = {0};
    size_t position = 0U;
    size_t ordinary_start = 0U;
    m3_status status;

    if (tokenizer == NULL || tokenizer->state == NULL || ids == NULL ||
        (input.length != 0U && input.data == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer encode argument is invalid");
    }
    status = m3_validate_utf8(input.data, input.length, error);
    while (status == M3_STATUS_OK && position < input.length) {
        const m3_added_token *added = m3_added_match(
            tokenizer->state, input.data + position, input.length - position);

        if (added != NULL) {
            status = m3_encode_ordinary(tokenizer->state,
                                        input.data + ordinary_start,
                                        position - ordinary_start,
                                        &built, error);
            if (status == M3_STATUS_OK) {
                status = m3_ids_append(&built, added->id, error);
            }
            position += added->content.length;
            ordinary_start = position;
        } else {
            uint32_t cp;
            size_t length;

            (void)m3_utf8_decode(input.data + position,
                                 input.length - position, &cp, &length);
            position += length;
        }
    }
    if (status == M3_STATUS_OK) {
        const uint8_t *ordinary = input.data == NULL
                                      ? NULL
                                      : input.data + ordinary_start;

        status = m3_encode_ordinary(tokenizer->state,
                                    ordinary,
                                    input.length - ordinary_start,
                                    &built, error);
    }
    if (status == M3_STATUS_OK) {
        m3_token_ids_dispose(ids);
        ids->data = built.data;
        ids->count = built.count;
        built.data = NULL;
        m3_error_reset(error);
    }
    free(built.data);
    return status;
}
