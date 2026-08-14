/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_internal.h"

#include "m3_unicode.h"

#include <stdlib.h>

static const m3_added_token *m3_decode_added(
    const m3_tokenizer_state *state, uint32_t id)
{
    size_t index;

    if (id < M3_TOKEN_END_OF_TEXT || id >= M3_TOKENIZER_ID_COUNT) {
        return NULL;
    }
    index = (size_t)(id - M3_TOKEN_END_OF_TEXT);
    return state->added[index].id == id ? &state->added[index] : NULL;
}

static m3_status m3_decode_raw(const m3_tokenizer_state *state,
                                const uint32_t *ids, size_t count,
                                bool skip_special, m3_byte_buffer *raw,
                                m3_error *error)
{
    size_t index;

    for (index = 0U; index < count; ++index) {
        uint32_t id = ids[index];
        const uint8_t *bytes;
        size_t length;

        if (id < state->model_vocab_count) {
            m3_token_span token = state->model_tokens[id];

            bytes = state->model_bytes + token.offset;
            length = token.length;
        } else {
            const m3_added_token *added = m3_decode_added(state, id);

            if (added == NULL) {
                if (id >= M3_TOKENIZER_SEMANTIC_FIRST &&
                    id <= M3_TOKENIZER_SEMANTIC_LAST) {
                    return m3_error_set(
                        error, M3_STATUS_OUT_OF_RANGE,
                        "semantic ID %u is not a tokenizer ID", id);
                }
                return m3_error_set(error, M3_STATUS_OUT_OF_RANGE,
                                    "token ID %u is invalid", id);
            }
            if (skip_special && added->special) {
                continue;
            }
            bytes = state->added_bytes + added->content.offset;
            length = added->content.length;
        }
        {
            m3_status status = m3_byte_buffer_append(raw, bytes, length,
                                                     error);

            if (status != M3_STATUS_OK) {
                return status;
            }
        }
    }
    return M3_STATUS_OK;
}

static m3_status m3_decode_utf8(const m3_byte_buffer *raw,
                                 m3_byte_buffer *decoded,
                                 m3_error *error)
{
    static const uint8_t replacement[] = {0xEFU, 0xBFU, 0xBDU};
    size_t position = 0U;
    m3_status status = M3_STATUS_OK;

    while (status == M3_STATUS_OK && position < raw->count) {
        uint32_t cp;
        size_t length;

        if (m3_utf8_decode(raw->data + position, raw->count - position,
                           &cp, &length)) {
            status = m3_byte_buffer_append(decoded, raw->data + position,
                                           length, error);
            position += length;
        } else {
            uint8_t first = raw->data[position];
            size_t remaining = raw->count - position;
            size_t invalid = 1U;
            size_t expected = 1U;
            bool second_ok = false;

            if (first >= 0xC2U && first <= 0xDFU) expected = 2U;
            else if (first >= 0xE0U && first <= 0xEFU) expected = 3U;
            else if (first >= 0xF0U && first <= 0xF4U) expected = 4U;
            if (expected > 1U && remaining >= 2U) {
                uint8_t second = raw->data[position + 1U];
                uint8_t minimum = first == 0xE0U ? 0xA0U
                                  : first == 0xF0U ? 0x90U
                                  : 0x80U;
                uint8_t maximum = first == 0xEDU ? 0x9FU
                                  : first == 0xF4U ? 0x8FU
                                  : 0xBFU;

                second_ok = second >= minimum && second <= maximum;
                if (second_ok) invalid = 2U;
            }
            if (second_ok && expected > 2U && remaining >= 3U &&
                (raw->data[position + 2U] & 0xC0U) == 0x80U) {
                invalid = 3U;
            }
            if (invalid == expected && remaining >= expected) {
                invalid = 1U;
            } else if (remaining < expected && invalid == remaining) {
                invalid = remaining;
            }
            status = m3_byte_buffer_append(decoded, replacement,
                                           sizeof(replacement), error);
            position += invalid;
        }
    }
    return status;
}

m3_status m3_tokenizer_decode(const m3_tokenizer *tokenizer,
                              const uint32_t *ids, size_t count,
                              bool skip_special, m3_tokenizer_text *text,
                              m3_error *error)
{
    m3_byte_buffer raw = {0};
    m3_byte_buffer decoded = {0};
    uint8_t terminator = 0U;
    m3_status status;

    if (tokenizer == NULL || tokenizer->state == NULL || text == NULL ||
        (count != 0U && ids == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer decode argument is invalid");
    }
    if (count > SIZE_MAX / sizeof(*ids) || count > SIZE_MAX / 3U) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "token decode count overflows");
    }
    status = m3_decode_raw(tokenizer->state, ids, count, skip_special,
                           &raw, error);
    if (status == M3_STATUS_OK) {
        status = m3_decode_utf8(&raw, &decoded, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_byte_buffer_append(&decoded, &terminator, 1U, error);
    }
    if (status == M3_STATUS_OK) {
        m3_tokenizer_text_dispose(text);
        text->data = decoded.data;
        text->length = decoded.count - 1U;
        decoded.data = NULL;
        m3_error_reset(error);
    }
    m3_byte_buffer_dispose(&raw);
    m3_byte_buffer_dispose(&decoded);
    return status;
}
