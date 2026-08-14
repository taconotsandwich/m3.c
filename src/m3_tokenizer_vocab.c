/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_model_internal.h"

#include "m3_unicode.h"

#include <stdlib.h>

static m3_status m3_vocab_decode_name(const m3_byte_buffer *names,
                                       m3_token_span name,
                                       m3_byte_buffer *raw,
                                       m3_token_span *decoded,
                                       m3_error *error)
{
    const uint8_t *data;
    size_t position = 0U;
    size_t start = raw->count;

    if (name.length == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "BPE vocabulary token is empty");
    }
    data = names->data + name.offset;
    while (position < name.length) {
        uint32_t cp;
        size_t length;
        uint8_t byte;
        m3_status status;

        if (!m3_utf8_decode(data + position, name.length - position,
                            &cp, &length) ||
            !m3_byte_alphabet_byte(cp, &byte)) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "BPE token is outside the byte alphabet");
        }
        status = m3_byte_buffer_append(raw, &byte, 1U, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        position += length;
    }
    if (start > UINT32_MAX || raw->count - start > UINT32_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "decoded BPE vocabulary is too large");
    }
    decoded->offset = (uint32_t)start;
    decoded->length = (uint32_t)(raw->count - start);
    return M3_STATUS_OK;
}

static m3_status m3_vocab_validate_alphabet(m3_tokenizer_build *build,
                                             m3_error *error)
{
    unsigned int byte_value;

    if (build == NULL || build->state == NULL ||
        build->state->model_bytes == NULL) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "BPE vocabulary storage is unavailable");
    }

    for (byte_value = 0U; byte_value <= 255U; ++byte_value) {
        uint8_t encoded[4];
        uint32_t cp = m3_byte_alphabet_code_point((uint8_t)byte_value);
        size_t length = m3_utf8_encode(cp, encoded);
        uint32_t id;
        m3_token_span token;

        if (!m3_tokenizer_name_lookup(build, encoded, length, &id) ||
            id >= 256U) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "BPE byte alphabet token is missing");
        }
        token = build->state->model_tokens[id];
        if (token.length != 1U ||
            build->state->model_bytes[token.offset] != (uint8_t)byte_value) {
            return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                "BPE byte alphabet mapping is inconsistent");
        }
        build->state->byte_ids[byte_value] = id;
    }
    return M3_STATUS_OK;
}

m3_status m3_tokenizer_parse_vocab(m3_tokenizer_build *build,
                                    const uint8_t *data,
                                    const m3_json_span *span,
                                    m3_error *error)
{
    m3_tokenizer_json json;
    m3_byte_buffer raw = {0};
    bool *seen_ids;
    uint32_t count = 0U;
    m3_status status;

    if (build == NULL || data == NULL || span == NULL || !span->present) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "BPE vocabulary argument is invalid");
    }
    seen_ids = calloc(build->contract->model_vocab_count, sizeof(*seen_ids));
    if (seen_ids == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate BPE vocabulary ID set");
    }
    m3_tokenizer_json_init(&json, data + span->start, span->size);
    status = m3_tokenizer_json_expect(&json, (uint8_t)'{', error);
    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
        m3_token_span name;
        m3_token_span decoded;
        uint32_t id;

        status = m3_tokenizer_json_string(&json, &build->name_bytes,
                                          &name, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(&json, (uint8_t)':', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_uint32(&json, &id, error);
        }
        if (status == M3_STATUS_OK &&
            (id >= build->contract->model_vocab_count || seen_ids[id])) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "BPE vocabulary ID is duplicate or invalid");
        }
        if (status == M3_STATUS_OK) {
            status = m3_vocab_decode_name(&build->name_bytes, name, &raw,
                                          &decoded, error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_name_insert(build, name, id, error);
        }
        if (status == M3_STATUS_OK) {
            seen_ids[id] = true;
            build->state->model_tokens[id] = decoded;
            count += 1U;
        }
        if (status == M3_STATUS_OK &&
            !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
            status = m3_tokenizer_json_separator(&json, (uint8_t)'}', error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(&json, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_finish(&json, error);
    }
    if (status == M3_STATUS_OK &&
        count != build->contract->model_vocab_count) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "BPE vocabulary has %u entries, expected %u",
                              count, build->contract->model_vocab_count);
    }
    if (status == M3_STATUS_OK) {
        build->state->model_bytes = raw.data;
        build->state->model_byte_count = raw.count;
        raw.data = NULL;
        raw.count = 0U;
        raw.capacity = 0U;
        status = m3_vocab_validate_alphabet(build, error);
    }
    free(seen_ids);
    m3_byte_buffer_dispose(&raw);
    return status;
}
