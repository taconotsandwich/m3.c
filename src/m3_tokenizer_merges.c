/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_model_internal.h"

#include <string.h>

static m3_status m3_merge_pair(m3_tokenizer_json *json,
                                m3_byte_buffer *names,
                                m3_token_span *left,
                                m3_token_span *right,
                                m3_error *error)
{
    m3_status status = m3_tokenizer_json_expect(json, (uint8_t)'[', error);

    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_string(json, names, left, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(json, (uint8_t)',', error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_string(json, names, right, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(json, (uint8_t)']', error);
    }
    return status;
}

static m3_status m3_merge_resolve(m3_tokenizer_build *build,
                                   const m3_byte_buffer *names,
                                   m3_token_span left_name,
                                   m3_token_span right_name,
                                   uint32_t rank, m3_error *error)
{
    m3_token_span combined;
    m3_token_span left_token;
    m3_token_span right_token;
    m3_token_span output_token;
    uint32_t left;
    uint32_t right;
    uint32_t output;
    uint32_t expected_output = 256U + rank;

    if (left_name.length == 0U || right_name.length == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "BPE merge token is empty");
    }
    if (left_name.offset > UINT32_MAX - left_name.length ||
        left_name.offset + left_name.length != right_name.offset ||
        left_name.length > UINT32_MAX - right_name.length) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "BPE merge name length overflows");
    }
    combined.offset = left_name.offset;
    combined.length = left_name.length + right_name.length;
    if (!m3_tokenizer_name_lookup(
            build, names->data + left_name.offset, left_name.length, &left) ||
        !m3_tokenizer_name_lookup(
            build, names->data + right_name.offset, right_name.length, &right) ||
        !m3_tokenizer_name_lookup(
            build, names->data + combined.offset, combined.length, &output)) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "BPE merge child or output token is missing");
    }
    if (output != expected_output || left >= output || right >= output) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "BPE merge rank mapping is inconsistent");
    }
    left_token = build->state->model_tokens[left];
    right_token = build->state->model_tokens[right];
    output_token = build->state->model_tokens[output];
    if (left_token.length > UINT32_MAX - right_token.length ||
        output_token.length != left_token.length + right_token.length ||
        memcmp(build->state->model_bytes + output_token.offset,
               build->state->model_bytes + left_token.offset,
               left_token.length) != 0 ||
        memcmp(build->state->model_bytes + output_token.offset +
                   left_token.length,
               build->state->model_bytes + right_token.offset,
               right_token.length) != 0) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "BPE merge output bytes are inconsistent");
    }
    return m3_tokenizer_merge_insert(build->state, left, right, rank,
                                     output, error);
}

m3_status m3_tokenizer_parse_merges(m3_tokenizer_build *build,
                                     const uint8_t *data,
                                     const m3_json_span *span,
                                     m3_error *error)
{
    m3_tokenizer_json json;
    m3_byte_buffer names = {0};
    uint32_t count = 0U;
    m3_status status;

    if (build == NULL || data == NULL || span == NULL || !span->present) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "BPE merge section argument is invalid");
    }
    m3_tokenizer_json_init(&json, data + span->start, span->size);
    status = m3_tokenizer_json_expect(&json, (uint8_t)'[', error);
    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(&json, (uint8_t)']')) {
        m3_token_span left;
        m3_token_span right;

        names.count = 0U;
        status = m3_merge_pair(&json, &names, &left, &right, error);
        if (status == M3_STATUS_OK &&
            count >= build->contract->merge_count) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "BPE merge list has too many entries");
        }
        if (status == M3_STATUS_OK) {
            status = m3_merge_resolve(build, &names, left, right, count,
                                      error);
        }
        if (status == M3_STATUS_OK) {
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
    if (status == M3_STATUS_OK && count != build->contract->merge_count) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "BPE merge list has %u entries, expected %u",
                              count, build->contract->merge_count);
    }
    m3_byte_buffer_dispose(&names);
    return status;
}
