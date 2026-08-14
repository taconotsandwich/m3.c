/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_model_internal.h"

#include <string.h>

static int m3_added_field(const m3_byte_buffer *buffer, m3_token_span key)
{
    static const char *const names[] = {
        "id", "content", "single_word", "lstrip", "rstrip", "normalized",
        "special",
    };
    size_t index;

    for (index = 0U; index < sizeof(names) / sizeof(names[0]); ++index) {
        if (m3_tokenizer_string_equal(buffer, key, names[index])) {
            return (int)index;
        }
    }
    return -1;
}

static m3_status m3_added_object(m3_tokenizer_json *json,
                                  m3_byte_buffer *contents,
                                  uint32_t *id, m3_token_span *content,
                                  bool *special, m3_error *error)
{
    enum { ID, CONTENT, SINGLE, LSTRIP, RSTRIP, NORMALIZED, SPECIAL,
           FIELD_COUNT };
    m3_byte_buffer scratch = {0};
    m3_token_span key;
    unsigned int seen = 0U;
    m3_status status = m3_tokenizer_json_expect(json, (uint8_t)'{', error);

    *id = UINT32_MAX;
    content->offset = 0U;
    content->length = 0U;
    *special = false;
    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
        int field;
        unsigned int bit;

        scratch.count = 0U;
        status = m3_tokenizer_json_string(json, &scratch, &key, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(json, (uint8_t)':', error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        field = m3_added_field(&scratch, key);
        if (field < 0) {
            const char *text = scratch.data == NULL
                                   ? ""
                                   : (const char *)scratch.data + key.offset;

            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "unknown added-token field '%.*s'",
                                  (int)key.length, text);
            break;
        }
        bit = 1U << (unsigned int)field;
        if ((seen & bit) != 0U) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "duplicate added-token field");
            break;
        }
        seen |= bit;
        if (field == ID) {
            status = m3_tokenizer_json_uint32(json, id, error);
        } else if (field == CONTENT) {
            status = m3_tokenizer_json_string(json, contents, content, error);
        } else {
            bool value = false;

            status = m3_tokenizer_json_bool(json, &value, error);
            if (status == M3_STATUS_OK && field == SPECIAL) {
                *special = value;
            } else if (status == M3_STATUS_OK && value) {
                status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                      "added-token matching flag must be false");
            }
        }
        if (status == M3_STATUS_OK &&
            !m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
            status = m3_tokenizer_json_separator(json, (uint8_t)'}', error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(json, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK && seen != (1U << FIELD_COUNT) - 1U) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "added-token object is incomplete");
    }
    m3_byte_buffer_dispose(&scratch);
    return status;
}

static bool m3_added_expected_special(uint32_t id)
{
    return (id >= 151643U && id <= 151656U) ||
           (id >= 151669U && id <= 151674U);
}

static m3_status m3_added_required(const m3_tokenizer_state *state,
                                    uint32_t id, const char *content,
                                    m3_error *error)
{
    const m3_added_token *token = &state->added[id - M3_TOKEN_END_OF_TEXT];
    size_t expected_length = strlen(content);

    if ((size_t)token->content.length != expected_length ||
        memcmp(state->added_bytes + token->content.offset, content,
               expected_length) != 0 || !token->special) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "required added token %u has wrong content", id);
    }
    return M3_STATUS_OK;
}

static m3_status m3_added_validate(const m3_tokenizer_state *state,
                                    m3_error *error)
{
    static const struct {
        uint32_t id;
        const char *content;
    } required[] = {
        {M3_TOKEN_END_OF_TEXT, "<|endoftext|>"},
        {M3_TOKEN_IM_START, "<|im_start|>"},
        {M3_TOKEN_IM_END, "<|im_end|>"},
        {M3_TOKEN_AUDIO_CFG, "<|audio_cfg|>"},
        {M3_TOKEN_AUDIO_START, "<|audio_start|>"},
        {M3_TOKEN_AUDIO_END, "<|audio_end|>"},
        {M3_TOKEN_CAPTION_START, "<|caption_start|>"},
        {M3_TOKEN_CAPTION_END, "<|caption_end|>"},
        {M3_TOKEN_LYRICS_START, "<|lyrics_start|>"},
        {M3_TOKEN_LYRICS_END, "<|lyrics_end|>"},
    };
    size_t left;
    size_t right;

    if (state == NULL || state->added_bytes == NULL) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "added-token storage is unavailable");
    }

    for (left = 0U; left < M3_TOKENIZER_ADDED_COUNT; ++left) {
        const m3_added_token *first = &state->added[left];

        if (first->content.length == 0U ||
            first->special != m3_added_expected_special(first->id)) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "added token %u has invalid attributes",
                                first->id);
        }
        for (right = left + 1U; right < M3_TOKENIZER_ADDED_COUNT; ++right) {
            const m3_added_token *second = &state->added[right];

            if (first->content.length == second->content.length &&
                memcmp(state->added_bytes + first->content.offset,
                       state->added_bytes + second->content.offset,
                       first->content.length) == 0) {
                return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                    "duplicate added-token content");
            }
        }
    }
    for (left = 0U; left < sizeof(required) / sizeof(required[0]); ++left) {
        m3_status status = m3_added_required(state, required[left].id,
                                             required[left].content, error);

        if (status != M3_STATUS_OK) {
            return status;
        }
    }
    return M3_STATUS_OK;
}

m3_status m3_tokenizer_parse_added(m3_tokenizer_build *build,
                                    const uint8_t *data,
                                    const m3_json_span *span,
                                    m3_error *error)
{
    m3_tokenizer_json json;
    m3_byte_buffer contents = {0};
    bool seen[M3_TOKENIZER_ADDED_COUNT] = {false};
    uint32_t count = 0U;
    m3_status status;

    if (build == NULL || data == NULL || span == NULL || !span->present) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "added-token section argument is invalid");
    }
    m3_tokenizer_json_init(&json, data + span->start, span->size);
    status = m3_tokenizer_json_expect(&json, (uint8_t)'[', error);
    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(&json, (uint8_t)']')) {
        uint32_t id;
        m3_token_span content;
        bool special;
        size_t index;

        status = m3_added_object(&json, &contents, &id, &content, &special,
                                 error);
        if (status == M3_STATUS_OK &&
            (id < M3_TOKEN_END_OF_TEXT || id >= M3_TOKENIZER_ID_COUNT)) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "added-token ID %u is invalid", id);
        }
        index = id >= M3_TOKEN_END_OF_TEXT
                    ? (size_t)(id - M3_TOKEN_END_OF_TEXT)
                    : M3_TOKENIZER_ADDED_COUNT;
        if (status == M3_STATUS_OK && seen[index]) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "duplicate added-token ID %u", id);
        }
        if (status == M3_STATUS_OK) {
            seen[index] = true;
            build->state->added[index].id = id;
            build->state->added[index].content = content;
            build->state->added[index].special = special;
            count += 1U;
        }
        if (status == M3_STATUS_OK &&
            !m3_tokenizer_json_next_is(&json, (uint8_t)']')) {
            status = m3_tokenizer_json_separator(&json, (uint8_t)']', error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(&json, (uint8_t)']', error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_finish(&json, error);
    }
    if (status == M3_STATUS_OK && count != M3_TOKENIZER_ADDED_COUNT) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "tokenizer has %u added tokens, expected %u",
                              count, M3_TOKENIZER_ADDED_COUNT);
    }
    if (status == M3_STATUS_OK) {
        build->state->added_bytes = contents.data;
        build->state->added_byte_count = contents.count;
        contents.data = NULL;
        contents.count = 0U;
        contents.capacity = 0U;
        status = m3_added_validate(build->state, error);
    }
    m3_byte_buffer_dispose(&contents);
    return status;
}
