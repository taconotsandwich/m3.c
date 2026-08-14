/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_internal.h"

#include <string.h>

static m3_status m3_schema_duplicate(m3_error *error, const char *field)
{
    return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                        "duplicate tokenizer field '%s'", field);
}

static m3_status m3_schema_unknown(m3_error *error,
                                   const m3_byte_buffer *scratch,
                                   m3_token_span key)
{
    const char *text = scratch->data == NULL
                           ? ""
                           : (const char *)scratch->data + key.offset;

    return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                        "unknown tokenizer field '%.*s'", (int)key.length,
                        text);
}

static m3_status m3_schema_string_value(m3_tokenizer_json *json,
                                         const char *expected,
                                         m3_error *error)
{
    m3_byte_buffer value = {0};
    m3_token_span span;
    m3_status status = m3_tokenizer_json_string(json, &value, &span, error);

    if (status == M3_STATUS_OK &&
        !m3_tokenizer_string_equal(&value, span, expected)) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "tokenizer string must be '%s'", expected);
    }
    m3_byte_buffer_dispose(&value);
    return status;
}

static m3_status m3_schema_span_reader(const uint8_t *data,
                                        const m3_json_span *span,
                                        m3_tokenizer_json *json,
                                        m3_error *error)
{
    if (data == NULL || span == NULL || json == NULL || !span->present) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "required tokenizer section is missing");
    }
    m3_tokenizer_json_init(json, data + span->start, span->size);
    return M3_STATUS_OK;
}

static m3_status m3_schema_null_span(const uint8_t *data,
                                      const m3_json_span *span,
                                      m3_error *error)
{
    m3_tokenizer_json json;
    m3_status status = m3_schema_span_reader(data, span, &json, error);

    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_null(&json, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_finish(&json, error);
    }
    return status;
}

static m3_status m3_schema_literal_span(const uint8_t *data,
                                         const m3_json_span *span,
                                         const char *expected,
                                         m3_error *error)
{
    m3_tokenizer_json json;
    m3_status status = m3_schema_span_reader(data, span, &json, error);

    if (status == M3_STATUS_OK) {
        status = m3_schema_string_value(&json, expected, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_finish(&json, error);
    }
    return status;
}

static m3_status m3_schema_normalizer(const uint8_t *data,
                                       const m3_json_span *span,
                                       m3_error *error)
{
    m3_tokenizer_json json;
    m3_byte_buffer scratch = {0};
    m3_token_span key = {0};
    bool seen_type = false;
    m3_status status = m3_schema_span_reader(data, span, &json, error);

    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(&json, (uint8_t)'{', error);
    }
    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
        scratch.count = 0U;
        status = m3_tokenizer_json_string(&json, &scratch, &key, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(&json, (uint8_t)':', error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        if (m3_tokenizer_string_equal(&scratch, key, "type")) {
            if (seen_type) {
                status = m3_schema_duplicate(error, "type");
            } else {
                seen_type = true;
                status = m3_schema_string_value(&json, "NFC", error);
            }
        } else {
            status = m3_schema_unknown(error, &scratch, key);
        }
        if (status == M3_STATUS_OK &&
            !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
            status = m3_tokenizer_json_separator(&json, (uint8_t)'}', error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(&json, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK && !seen_type) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "NFC normalizer type is missing");
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_finish(&json, error);
    }
    m3_byte_buffer_dispose(&scratch);
    return status;
}

static m3_status m3_schema_byte_level(m3_tokenizer_json *json,
                                       bool expected_prefix,
                                       bool expected_trim,
                                       bool expected_regex,
                                       m3_error *error)
{
    m3_byte_buffer scratch = {0};
    m3_token_span key;
    unsigned int seen = 0U;
    m3_status status = m3_tokenizer_json_expect(json, (uint8_t)'{', error);

    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
        bool value = false;
        unsigned int bit = 0U;
        bool expected = false;

        scratch.count = 0U;
        status = m3_tokenizer_json_string(json, &scratch, &key, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(json, (uint8_t)':', error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        if (m3_tokenizer_string_equal(&scratch, key, "type")) {
            bit = 1U;
            if ((seen & bit) == 0U) {
                status = m3_schema_string_value(json, "ByteLevel", error);
            }
        } else if (m3_tokenizer_string_equal(&scratch, key,
                                              "add_prefix_space")) {
            bit = 2U;
            expected = expected_prefix;
            if ((seen & bit) == 0U) {
                status = m3_tokenizer_json_bool(json, &value, error);
            }
        } else if (m3_tokenizer_string_equal(&scratch, key,
                                              "trim_offsets")) {
            bit = 4U;
            expected = expected_trim;
            if ((seen & bit) == 0U) {
                status = m3_tokenizer_json_bool(json, &value, error);
            }
        } else if (m3_tokenizer_string_equal(&scratch, key,
                                              "use_regex")) {
            bit = 8U;
            expected = expected_regex;
            if ((seen & bit) == 0U) {
                status = m3_tokenizer_json_bool(json, &value, error);
            }
        } else {
            status = m3_schema_unknown(error, &scratch, key);
        }
        if (status == M3_STATUS_OK && bit != 0U && (seen & bit) != 0U) {
            status = m3_schema_duplicate(error, "ByteLevel field");
        }
        if (status == M3_STATUS_OK && bit > 1U && value != expected) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "ByteLevel Boolean has wrong value");
        }
        seen |= bit;
        if (status == M3_STATUS_OK &&
            !m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
            status = m3_tokenizer_json_separator(json, (uint8_t)'}', error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(json, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK && seen != 15U) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "ByteLevel object is incomplete");
    }
    m3_byte_buffer_dispose(&scratch);
    return status;
}

static m3_status m3_schema_pattern(m3_tokenizer_json *json,
                                    m3_error *error)
{
    static const char expression[] =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
        "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
        "\\s+(?!\\S)|\\s+";
    m3_byte_buffer scratch = {0};
    m3_token_span key = {0};
    bool seen = false;
    m3_status status = m3_tokenizer_json_expect(json, (uint8_t)'{', error);

    if (status == M3_STATUS_OK &&
        m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "Split Regex is missing");
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_string(json, &scratch, &key, error);
    }
    if (status == M3_STATUS_OK &&
        !m3_tokenizer_string_equal(&scratch, key, "Regex")) {
        status = m3_schema_unknown(error, &scratch, key);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(json, (uint8_t)':', error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_schema_string_value(json, expression, error);
        seen = status == M3_STATUS_OK;
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(json, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK && !seen) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "Split Regex is missing");
    }
    m3_byte_buffer_dispose(&scratch);
    return status;
}

static m3_status m3_schema_split(m3_tokenizer_json *json, m3_error *error)
{
    m3_byte_buffer scratch = {0};
    m3_token_span key;
    unsigned int seen = 0U;
    m3_status status = m3_tokenizer_json_expect(json, (uint8_t)'{', error);

    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
        unsigned int bit = 0U;

        scratch.count = 0U;
        status = m3_tokenizer_json_string(json, &scratch, &key, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(json, (uint8_t)':', error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        if (m3_tokenizer_string_equal(&scratch, key, "type")) {
            bit = 1U;
            if ((seen & bit) == 0U) {
                status = m3_schema_string_value(json, "Split", error);
            }
        } else if (m3_tokenizer_string_equal(&scratch, key, "pattern")) {
            bit = 2U;
            if ((seen & bit) == 0U) {
                status = m3_schema_pattern(json, error);
            }
        } else if (m3_tokenizer_string_equal(&scratch, key, "behavior")) {
            bit = 4U;
            if ((seen & bit) == 0U) {
                status = m3_schema_string_value(json, "Isolated", error);
            }
        } else if (m3_tokenizer_string_equal(&scratch, key, "invert")) {
            bool value = true;
            bit = 8U;
            if ((seen & bit) == 0U) {
                status = m3_tokenizer_json_bool(json, &value, error);
            }
            if (status == M3_STATUS_OK && value) {
                status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                      "Split invert must be false");
            }
        } else {
            status = m3_schema_unknown(error, &scratch, key);
        }
        if (status == M3_STATUS_OK && bit != 0U && (seen & bit) != 0U) {
            status = m3_schema_duplicate(error, "Split field");
        }
        seen |= bit;
        if (status == M3_STATUS_OK &&
            !m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
            status = m3_tokenizer_json_separator(json, (uint8_t)'}', error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(json, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK && seen != 15U) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "Split object is incomplete");
    }
    m3_byte_buffer_dispose(&scratch);
    return status;
}

static m3_status m3_schema_pretokenizer(const uint8_t *data,
                                         const m3_json_span *span,
                                         m3_error *error)
{
    m3_tokenizer_json json;
    m3_byte_buffer scratch = {0};
    m3_token_span key;
    unsigned int seen = 0U;
    m3_status status = m3_schema_span_reader(data, span, &json, error);

    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(&json, (uint8_t)'{', error);
    }
    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
        unsigned int bit = 0U;

        scratch.count = 0U;
        status = m3_tokenizer_json_string(&json, &scratch, &key, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(&json, (uint8_t)':', error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        if (m3_tokenizer_string_equal(&scratch, key, "type")) {
            bit = 1U;
            if ((seen & bit) == 0U) {
                status = m3_schema_string_value(&json, "Sequence", error);
            }
        } else if (m3_tokenizer_string_equal(&scratch, key,
                                              "pretokenizers")) {
            bit = 2U;
            if ((seen & bit) == 0U) {
                status = m3_tokenizer_json_expect(&json, (uint8_t)'[', error);
                if (status == M3_STATUS_OK) {
                    status = m3_schema_split(&json, error);
                }
                if (status == M3_STATUS_OK) {
                    status = m3_tokenizer_json_expect(&json, (uint8_t)',', error);
                }
                if (status == M3_STATUS_OK) {
                    status = m3_schema_byte_level(&json, false, true, false,
                                                  error);
                }
                if (status == M3_STATUS_OK) {
                    status = m3_tokenizer_json_expect(&json, (uint8_t)']', error);
                }
            }
        } else {
            status = m3_schema_unknown(error, &scratch, key);
        }
        if (status == M3_STATUS_OK && bit != 0U && (seen & bit) != 0U) {
            status = m3_schema_duplicate(error, "pre-tokenizer field");
        }
        seen |= bit;
        if (status == M3_STATUS_OK &&
            !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
            status = m3_tokenizer_json_separator(&json, (uint8_t)'}', error);
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_expect(&json, (uint8_t)'}', error);
    }
    if (status == M3_STATUS_OK && seen != 3U) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "pre-tokenizer object is incomplete");
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_finish(&json, error);
    }
    m3_byte_buffer_dispose(&scratch);
    return status;
}

static m3_status m3_schema_byte_level_span(const uint8_t *data,
                                            const m3_json_span *span,
                                            bool prefix, bool trim,
                                            bool regex, m3_error *error)
{
    m3_tokenizer_json json;
    m3_status status = m3_schema_span_reader(data, span, &json, error);

    if (status == M3_STATUS_OK) {
        status = m3_schema_byte_level(&json, prefix, trim, regex, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_tokenizer_json_finish(&json, error);
    }
    return status;
}

m3_status m3_tokenizer_parse_sections(const uint8_t *data, size_t size,
                                       m3_tokenizer_sections *sections,
                                       m3_error *error)
{
    enum { VERSION, TRUNCATION, PADDING, ADDED, NORMALIZER, PRETOKENIZER,
           POSTPROCESSOR, DECODER, MODEL, FIELD_COUNT };
    m3_tokenizer_json json;
    m3_json_span spans[FIELD_COUNT] = {{0}};
    m3_byte_buffer scratch = {0};
    m3_token_span key;
    unsigned int seen = 0U;
    m3_status status;

    if (data == NULL || size == 0U || size > M3_TOKENIZER_JSON_MAX_BYTES ||
        sections == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer JSON input is invalid");
    }
    (void)memset(sections, 0, sizeof(*sections));
    m3_tokenizer_json_init(&json, data, size);
    status = m3_tokenizer_json_expect(&json, (uint8_t)'{', error);
    while (status == M3_STATUS_OK &&
           !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
        int field = -1;
        unsigned int bit;

        scratch.count = 0U;
        status = m3_tokenizer_json_string(&json, &scratch, &key, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(&json, (uint8_t)':', error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        if (m3_tokenizer_string_equal(&scratch, key, "version")) field = VERSION;
        else if (m3_tokenizer_string_equal(&scratch, key, "truncation")) field = TRUNCATION;
        else if (m3_tokenizer_string_equal(&scratch, key, "padding")) field = PADDING;
        else if (m3_tokenizer_string_equal(&scratch, key, "added_tokens")) field = ADDED;
        else if (m3_tokenizer_string_equal(&scratch, key, "normalizer")) field = NORMALIZER;
        else if (m3_tokenizer_string_equal(&scratch, key, "pre_tokenizer")) field = PRETOKENIZER;
        else if (m3_tokenizer_string_equal(&scratch, key, "post_processor")) field = POSTPROCESSOR;
        else if (m3_tokenizer_string_equal(&scratch, key, "decoder")) field = DECODER;
        else if (m3_tokenizer_string_equal(&scratch, key, "model")) field = MODEL;
        else status = m3_schema_unknown(error, &scratch, key);
        if (status != M3_STATUS_OK || field < 0) break;
        bit = 1U << (unsigned int)field;
        if ((seen & bit) != 0U) {
            status = m3_schema_duplicate(error, "root field");
        } else {
            seen |= bit;
            status = m3_tokenizer_json_span(&json, &spans[field], error);
        }
        if (status == M3_STATUS_OK &&
            !m3_tokenizer_json_next_is(&json, (uint8_t)'}')) {
            status = m3_tokenizer_json_separator(&json, (uint8_t)'}', error);
        }
    }
    if (status == M3_STATUS_OK) status = m3_tokenizer_json_expect(&json, (uint8_t)'}', error);
    if (status == M3_STATUS_OK) status = m3_tokenizer_json_finish(&json, error);
    if (status == M3_STATUS_OK && seen != (1U << FIELD_COUNT) - 1U) {
        status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                              "tokenizer root object is incomplete");
    }
    if (status == M3_STATUS_OK) status = m3_schema_literal_span(data, &spans[VERSION], "1.0", error);
    if (status == M3_STATUS_OK) status = m3_schema_null_span(data, &spans[TRUNCATION], error);
    if (status == M3_STATUS_OK) status = m3_schema_null_span(data, &spans[PADDING], error);
    if (status == M3_STATUS_OK) status = m3_schema_normalizer(data, &spans[NORMALIZER], error);
    if (status == M3_STATUS_OK) status = m3_schema_pretokenizer(data, &spans[PRETOKENIZER], error);
    if (status == M3_STATUS_OK) status = m3_schema_byte_level_span(data, &spans[POSTPROCESSOR], false, false, false, error);
    if (status == M3_STATUS_OK) status = m3_schema_byte_level_span(data, &spans[DECODER], true, true, true, error);
    if (status == M3_STATUS_OK) {
        sections->added_tokens = spans[ADDED];
        sections->model = spans[MODEL];
    }
    m3_byte_buffer_dispose(&scratch);
    return status;
}
