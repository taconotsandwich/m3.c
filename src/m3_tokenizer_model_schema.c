/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_internal.h"

static m3_status m3_model_string(m3_tokenizer_json *json,
                                  const char *expected, m3_error *error)
{
    m3_byte_buffer buffer = {0};
    m3_token_span value;
    m3_status status = m3_tokenizer_json_string(json, &buffer, &value, error);

    if (status == M3_STATUS_OK &&
        !m3_tokenizer_string_equal(&buffer, value, expected)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "BPE model string has wrong value");
    }
    m3_byte_buffer_dispose(&buffer);
    return status;
}

static int m3_model_field(const m3_byte_buffer *buffer, m3_token_span key)
{
    static const char *const names[] = {
        "type", "dropout", "unk_token", "continuing_subword_prefix",
        "end_of_word_suffix", "fuse_unk", "byte_fallback",
        "ignore_merges", "vocab", "merges",
    };
    size_t index;

    for (index = 0U; index < sizeof(names) / sizeof(names[0]); ++index) {
        if (m3_tokenizer_string_equal(buffer, key, names[index])) {
            return (int)index;
        }
    }
    return -1;
}

m3_status m3_tokenizer_parse_model_sections(
    const uint8_t *data, const m3_json_span *span,
    const m3_tokenizer_contract *contract, m3_model_sections *sections,
    m3_error *error)
{
    enum { TYPE, DROPOUT, UNK, PREFIX, SUFFIX, FUSE, FALLBACK, IGNORE,
           VOCAB, MERGES, FIELD_COUNT };
    m3_tokenizer_json json;
    m3_byte_buffer scratch = {0};
    m3_token_span key;
    unsigned int seen = 0U;
    m3_status status;

    if (data == NULL || span == NULL || !span->present || contract == NULL ||
        sections == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "BPE model section argument is invalid");
    }
    sections->vocab.present = false;
    sections->merges.present = false;
    m3_tokenizer_json_init(&json, data + span->start, span->size);
    status = m3_tokenizer_json_expect(&json, (uint8_t)'{', error);
    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
        int field;
        unsigned int bit;

        scratch.count = 0U;
        status = m3_tokenizer_json_string(&json, &scratch, &key, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(&json, (uint8_t)':', error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        field = m3_model_field(&scratch, key);
        if (field < 0) {
            const char *text = scratch.data == NULL
                                   ? ""
                                   : (const char *)scratch.data + key.offset;

            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "unknown BPE model field '%.*s'",
                                  (int)key.length, text);
            break;
        }
        bit = 1U << (unsigned int)field;
        if ((seen & bit) != 0U) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "duplicate BPE model field");
            break;
        }
        seen |= bit;
        switch (field) {
        case TYPE:
            status = m3_model_string(&json, "BPE", error);
            break;
        case DROPOUT:
        case UNK:
            status = m3_tokenizer_json_null(&json, error);
            break;
        case PREFIX:
        case SUFFIX:
            status = m3_model_string(&json, "", error);
            break;
        case FUSE:
        case FALLBACK:
        case IGNORE: {
            bool value = true;
            status = m3_tokenizer_json_bool(&json, &value, error);
            if (status == M3_STATUS_OK && value) {
                status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                      "BPE model flag must be false");
            }
            break;
        }
        case VOCAB:
            status = m3_tokenizer_json_span(&json, &sections->vocab, error);
            break;
        case MERGES:
            status = m3_tokenizer_json_span(&json, &sections->merges, error);
            break;
        default:
            status = m3_error_set(error, M3_STATUS_INTERNAL,
                                  "invalid BPE model field dispatch");
            break;
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
    if (status == M3_STATUS_OK && seen != (1U << FIELD_COUNT) - 1U) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "BPE model object is incomplete");
    }
    if (status == M3_STATUS_OK) {
        sections->vocab.start += span->start;
        sections->merges.start += span->start;
    }
    m3_byte_buffer_dispose(&scratch);
    return status;
}
