/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_internal.h"

#include "m3_unicode.h"

#include <limits.h>
#include <string.h>

static void m3_tjson_whitespace(m3_tokenizer_json *json)
{
    while (json->position < json->size) {
        uint8_t byte = json->data[json->position];

        if (byte != (uint8_t)' ' && byte != (uint8_t)'\t' &&
            byte != (uint8_t)'\r' && byte != (uint8_t)'\n') {
            break;
        }
        json->position += 1U;
    }
}

static m3_status m3_tjson_invalid(const m3_tokenizer_json *json,
                                  m3_error *error, const char *message)
{
    return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                        "tokenizer JSON byte %zu: %s", json->position,
                        message);
}

void m3_tokenizer_json_init(m3_tokenizer_json *json, const uint8_t *data,
                            size_t size)
{
    if (json != NULL) {
        json->data = data;
        json->size = size;
        json->position = 0U;
        json->depth = 0U;
    }
}

bool m3_tokenizer_json_next_is(m3_tokenizer_json *json, uint8_t byte)
{
    if (json == NULL) {
        return false;
    }
    m3_tjson_whitespace(json);
    return json->position < json->size && json->data[json->position] == byte;
}

m3_status m3_tokenizer_json_expect(m3_tokenizer_json *json, uint8_t byte,
                                    m3_error *error)
{
    if (json == NULL || json->data == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer JSON reader is null");
    }
    m3_tjson_whitespace(json);
    if (json->position >= json->size || json->data[json->position] != byte) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "tokenizer JSON byte %zu: expected '%c'",
                            json->position, (int)byte);
    }
    json->position += 1U;
    return M3_STATUS_OK;
}

m3_status m3_tokenizer_json_separator(m3_tokenizer_json *json,
                                       uint8_t closing, m3_error *error)
{
    m3_status status = m3_tokenizer_json_expect(json, (uint8_t)',', error);

    if (status == M3_STATUS_OK && m3_tokenizer_json_next_is(json, closing)) {
        return m3_tjson_invalid(json, error, "trailing comma");
    }
    return status;
}

static int m3_tjson_hex(uint8_t byte)
{
    if (byte >= (uint8_t)'0' && byte <= (uint8_t)'9') {
        return (int)(byte - (uint8_t)'0');
    }
    if (byte >= (uint8_t)'a' && byte <= (uint8_t)'f') {
        return 10 + (int)(byte - (uint8_t)'a');
    }
    if (byte >= (uint8_t)'A' && byte <= (uint8_t)'F') {
        return 10 + (int)(byte - (uint8_t)'A');
    }
    return -1;
}

static bool m3_tjson_hex_quad(m3_tokenizer_json *json, uint32_t *value)
{
    uint32_t result = 0U;
    unsigned int index;

    if (json->size - json->position < 4U) {
        return false;
    }
    for (index = 0U; index < 4U; ++index) {
        int digit = m3_tjson_hex(json->data[json->position + index]);

        if (digit < 0) {
            return false;
        }
        result = (result << 4U) | (uint32_t)digit;
    }
    json->position += 4U;
    *value = result;
    return true;
}

static m3_status m3_tjson_append_scalar(m3_byte_buffer *buffer, uint32_t cp,
                                        m3_error *error)
{
    uint8_t encoded[4];
    size_t length;

    if (cp == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "tokenizer JSON strings cannot contain NUL");
    }
    length = m3_utf8_encode(cp, encoded);
    if (length == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "tokenizer JSON has invalid Unicode scalar");
    }
    return m3_byte_buffer_append(buffer, encoded, length, error);
}

static m3_status m3_tjson_escape(m3_tokenizer_json *json,
                                  m3_byte_buffer *buffer, m3_error *error)
{
    uint8_t escape;
    uint8_t value;
    uint32_t cp;

    if (json->position >= json->size) {
        return m3_tjson_invalid(json, error, "unterminated escape");
    }
    escape = json->data[json->position++];
    switch (escape) {
    case (uint8_t)'"':
    case (uint8_t)'\\':
    case (uint8_t)'/':
        value = escape;
        return m3_byte_buffer_append(buffer, &value, 1U, error);
    case (uint8_t)'b': value = (uint8_t)'\b'; break;
    case (uint8_t)'f': value = (uint8_t)'\f'; break;
    case (uint8_t)'n': value = (uint8_t)'\n'; break;
    case (uint8_t)'r': value = (uint8_t)'\r'; break;
    case (uint8_t)'t': value = (uint8_t)'\t'; break;
    case (uint8_t)'u':
        if (!m3_tjson_hex_quad(json, &cp)) {
            return m3_tjson_invalid(json, error, "invalid Unicode escape");
        }
        if (cp >= 0xD800U && cp <= 0xDBFFU) {
            uint32_t low;

            if (json->size - json->position < 6U ||
                json->data[json->position] != (uint8_t)'\\' ||
                json->data[json->position + 1U] != (uint8_t)'u') {
                return m3_tjson_invalid(json, error,
                                        "missing low Unicode surrogate");
            }
            json->position += 2U;
            if (!m3_tjson_hex_quad(json, &low) || low < 0xDC00U ||
                low > 0xDFFFU) {
                return m3_tjson_invalid(json, error,
                                        "invalid low Unicode surrogate");
            }
            cp = 0x10000U + ((cp - 0xD800U) << 10U) + (low - 0xDC00U);
        } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
            return m3_tjson_invalid(json, error,
                                    "unexpected low Unicode surrogate");
        }
        return m3_tjson_append_scalar(buffer, cp, error);
    default:
        return m3_tjson_invalid(json, error, "invalid escape");
    }
    return m3_byte_buffer_append(buffer, &value, 1U, error);
}

m3_status m3_tokenizer_json_string(m3_tokenizer_json *json,
                                    m3_byte_buffer *buffer,
                                    m3_token_span *span, m3_error *error)
{
    size_t start;
    m3_status status;

    if (buffer == NULL || span == NULL || buffer->count > UINT32_MAX) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer JSON string destination is invalid");
    }
    status = m3_tokenizer_json_expect(json, (uint8_t)'"', error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    start = buffer->count;
    while (json->position < json->size) {
        uint8_t byte = json->data[json->position++];

        if (byte == (uint8_t)'"') {
            size_t length = buffer->count - start;

            if (start > UINT32_MAX || length > UINT32_MAX) {
                return m3_error_set(error, M3_STATUS_OVERFLOW,
                                    "tokenizer JSON string is too large");
            }
            span->offset = (uint32_t)start;
            span->length = (uint32_t)length;
            return M3_STATUS_OK;
        }
        if (byte < 0x20U) {
            return m3_tjson_invalid(json, error,
                                    "control byte in string");
        }
        if (byte == (uint8_t)'\\') {
            status = m3_tjson_escape(json, buffer, error);
            if (status != M3_STATUS_OK) {
                return status;
            }
        } else if (byte < 0x80U) {
            status = m3_byte_buffer_append(buffer, &byte, 1U, error);
            if (status != M3_STATUS_OK) {
                return status;
            }
        } else {
            uint32_t cp;
            size_t length;
            size_t scalar_start = json->position - 1U;

            if (!m3_utf8_decode(json->data + scalar_start,
                                json->size - scalar_start, &cp, &length)) {
                return m3_tjson_invalid(json, error,
                                        "invalid UTF-8 in string");
            }
            status = m3_byte_buffer_append(buffer,
                                           json->data + scalar_start,
                                           length, error);
            if (status != M3_STATUS_OK) {
                return status;
            }
            json->position = scalar_start + length;
        }
    }
    return m3_tjson_invalid(json, error, "unterminated string");
}

static bool m3_tjson_literal(m3_tokenizer_json *json, const char *literal)
{
    size_t length = strlen(literal);

    m3_tjson_whitespace(json);
    if (length > json->size - json->position ||
        memcmp(json->data + json->position, literal, length) != 0) {
        return false;
    }
    json->position += length;
    return true;
}

m3_status m3_tokenizer_json_bool(m3_tokenizer_json *json, bool *value,
                                 m3_error *error)
{
    if (json == NULL || value == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer JSON Boolean output is null");
    }
    if (m3_tjson_literal(json, "true")) {
        *value = true;
        return M3_STATUS_OK;
    }
    if (m3_tjson_literal(json, "false")) {
        *value = false;
        return M3_STATUS_OK;
    }
    return m3_tjson_invalid(json, error, "expected Boolean");
}

m3_status m3_tokenizer_json_null(m3_tokenizer_json *json, m3_error *error)
{
    if (json == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer JSON reader is null");
    }
    if (!m3_tjson_literal(json, "null")) {
        return m3_tjson_invalid(json, error, "expected null");
    }
    return M3_STATUS_OK;
}

m3_status m3_tokenizer_json_uint32(m3_tokenizer_json *json, uint32_t *value,
                                    m3_error *error)
{
    uint32_t result = 0U;
    size_t start;

    if (json == NULL || value == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer JSON integer output is null");
    }
    m3_tjson_whitespace(json);
    start = json->position;
    if (start >= json->size || json->data[start] < (uint8_t)'0' ||
        json->data[start] > (uint8_t)'9') {
        return m3_tjson_invalid(json, error, "expected unsigned integer");
    }
    if (json->data[start] == (uint8_t)'0' && start + 1U < json->size &&
        json->data[start + 1U] >= (uint8_t)'0' &&
        json->data[start + 1U] <= (uint8_t)'9') {
        return m3_tjson_invalid(json, error, "leading zero in integer");
    }
    while (json->position < json->size &&
           json->data[json->position] >= (uint8_t)'0' &&
           json->data[json->position] <= (uint8_t)'9') {
        uint32_t digit = json->data[json->position] - (uint8_t)'0';

        if (result > (UINT32_MAX - digit) / 10U) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "tokenizer JSON integer overflows");
        }
        result = result * 10U + digit;
        json->position += 1U;
    }
    *value = result;
    return M3_STATUS_OK;
}

static m3_status m3_tjson_skip_string(m3_tokenizer_json *json,
                                      m3_error *error)
{
    m3_byte_buffer discard = {0};
    m3_token_span span;
    m3_status status = m3_tokenizer_json_string(json, &discard, &span, error);

    m3_byte_buffer_dispose(&discard);
    return status;
}

static m3_status m3_tjson_skip_number(m3_tokenizer_json *json,
                                      m3_error *error)
{
    size_t start;

    m3_tjson_whitespace(json);
    start = json->position;
    if (json->position < json->size &&
        json->data[json->position] == (uint8_t)'-') {
        json->position += 1U;
    }
    if (json->position >= json->size) {
        return m3_tjson_invalid(json, error, "invalid number");
    }
    if (json->data[json->position] == (uint8_t)'0') {
        json->position += 1U;
    } else if (json->data[json->position] >= (uint8_t)'1' &&
               json->data[json->position] <= (uint8_t)'9') {
        while (json->position < json->size &&
               json->data[json->position] >= (uint8_t)'0' &&
               json->data[json->position] <= (uint8_t)'9') {
            json->position += 1U;
        }
    } else {
        return m3_tjson_invalid(json, error, "invalid number");
    }
    if (json->position < json->size &&
        json->data[json->position] == (uint8_t)'.') {
        json->position += 1U;
        if (json->position >= json->size ||
            json->data[json->position] < (uint8_t)'0' ||
            json->data[json->position] > (uint8_t)'9') {
            return m3_tjson_invalid(json, error, "invalid fraction");
        }
        while (json->position < json->size &&
               json->data[json->position] >= (uint8_t)'0' &&
               json->data[json->position] <= (uint8_t)'9') {
            json->position += 1U;
        }
    }
    if (json->position < json->size &&
        (json->data[json->position] == (uint8_t)'e' ||
         json->data[json->position] == (uint8_t)'E')) {
        json->position += 1U;
        if (json->position < json->size &&
            (json->data[json->position] == (uint8_t)'+' ||
             json->data[json->position] == (uint8_t)'-')) {
            json->position += 1U;
        }
        if (json->position >= json->size ||
            json->data[json->position] < (uint8_t)'0' ||
            json->data[json->position] > (uint8_t)'9') {
            return m3_tjson_invalid(json, error, "invalid exponent");
        }
        while (json->position < json->size &&
               json->data[json->position] >= (uint8_t)'0' &&
               json->data[json->position] <= (uint8_t)'9') {
            json->position += 1U;
        }
    }
    return json->position == start
               ? m3_tjson_invalid(json, error, "invalid number")
               : M3_STATUS_OK;
}

static m3_status m3_tjson_skip_value(m3_tokenizer_json *json,
                                     m3_error *error);

static m3_status m3_tjson_skip_array(m3_tokenizer_json *json,
                                     m3_error *error)
{
    m3_status status = m3_tokenizer_json_expect(json, (uint8_t)'[', error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (m3_tokenizer_json_next_is(json, (uint8_t)']')) {
        return m3_tokenizer_json_expect(json, (uint8_t)']', error);
    }
    for (;;) {
        status = m3_tjson_skip_value(json, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (m3_tokenizer_json_next_is(json, (uint8_t)']')) {
            return m3_tokenizer_json_expect(json, (uint8_t)']', error);
        }
        status = m3_tokenizer_json_expect(json, (uint8_t)',', error);
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
}

static m3_status m3_tjson_skip_object(m3_tokenizer_json *json,
                                      m3_error *error)
{
    m3_status status = m3_tokenizer_json_expect(json, (uint8_t)'{', error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    if (m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
        return m3_tokenizer_json_expect(json, (uint8_t)'}', error);
    }
    for (;;) {
        status = m3_tjson_skip_string(json, error);
        if (status == M3_STATUS_OK) {
            status = m3_tokenizer_json_expect(json, (uint8_t)':', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_tjson_skip_value(json, error);
        }
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (m3_tokenizer_json_next_is(json, (uint8_t)'}')) {
            return m3_tokenizer_json_expect(json, (uint8_t)'}', error);
        }
        status = m3_tokenizer_json_expect(json, (uint8_t)',', error);
        if (status != M3_STATUS_OK) {
            return status;
        }
    }
}

static m3_status m3_tjson_skip_value(m3_tokenizer_json *json,
                                     m3_error *error)
{
    m3_status status;

    if (json->depth >= 64U) {
        return m3_tjson_invalid(json, error, "nesting is too deep");
    }
    m3_tjson_whitespace(json);
    if (json->position >= json->size) {
        return m3_tjson_invalid(json, error, "missing value");
    }
    json->depth += 1U;
    switch (json->data[json->position]) {
    case (uint8_t)'"': status = m3_tjson_skip_string(json, error); break;
    case (uint8_t)'[': status = m3_tjson_skip_array(json, error); break;
    case (uint8_t)'{': status = m3_tjson_skip_object(json, error); break;
    case (uint8_t)'t':
        status = m3_tjson_literal(json, "true") ? M3_STATUS_OK
                 : m3_tjson_invalid(json, error, "invalid literal");
        break;
    case (uint8_t)'f':
        status = m3_tjson_literal(json, "false") ? M3_STATUS_OK
                 : m3_tjson_invalid(json, error, "invalid literal");
        break;
    case (uint8_t)'n':
        status = m3_tjson_literal(json, "null") ? M3_STATUS_OK
                 : m3_tjson_invalid(json, error, "invalid literal");
        break;
    default: status = m3_tjson_skip_number(json, error); break;
    }
    json->depth -= 1U;
    return status;
}

m3_status m3_tokenizer_json_span(m3_tokenizer_json *json, m3_json_span *span,
                                 m3_error *error)
{
    size_t start;
    m3_status status;

    if (json == NULL || span == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer JSON span argument is null");
    }
    m3_tjson_whitespace(json);
    start = json->position;
    status = m3_tjson_skip_value(json, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    span->start = start;
    span->size = json->position - start;
    span->present = true;
    return M3_STATUS_OK;
}

m3_status m3_tokenizer_json_finish(m3_tokenizer_json *json, m3_error *error)
{
    if (json == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "tokenizer JSON reader is null");
    }
    m3_tjson_whitespace(json);
    if (json->position != json->size) {
        return m3_tjson_invalid(json, error, "trailing data");
    }
    return M3_STATUS_OK;
}
