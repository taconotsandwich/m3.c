/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_json.h"

#include <stdlib.h>

static void m3_json_skip_whitespace(m3_json_reader *reader)
{
    while (reader->position < reader->size) {
        uint8_t byte = reader->data[reader->position];

        if (byte != (uint8_t)' ' && byte != (uint8_t)'\t' &&
            byte != (uint8_t)'\n' && byte != (uint8_t)'\r') {
            break;
        }
        reader->position += 1U;
    }
}

static m3_status m3_json_invalid(const m3_json_reader *reader,
                                 m3_error *error, const char *message)
{
    return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                        "JSON byte %zu: %s", reader->position, message);
}

void m3_json_reader_init(m3_json_reader *reader, const uint8_t *data,
                         size_t size)
{
    if (reader == NULL) {
        return;
    }
    reader->data = data;
    reader->size = size;
    reader->position = 0U;
    reader->depth = 0U;
}

bool m3_json_next_is(m3_json_reader *reader, uint8_t byte)
{
    if (reader == NULL) {
        return false;
    }
    m3_json_skip_whitespace(reader);
    return reader->position < reader->size &&
           reader->data[reader->position] == byte;
}

m3_status m3_json_expect(m3_json_reader *reader, uint8_t byte,
                         m3_error *error)
{
    if (reader == NULL || reader->data == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "JSON reader is null");
    }
    m3_json_skip_whitespace(reader);
    if (reader->position >= reader->size ||
        reader->data[reader->position] != byte) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "JSON byte %zu: expected '%c'", reader->position,
                            (int)byte);
    }
    reader->position += 1U;
    return M3_STATUS_OK;
}

static int m3_json_hex_value(uint8_t byte)
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

static bool m3_json_read_hex_quad(m3_json_reader *reader,
                                  uint32_t *code_point)
{
    uint32_t result = 0U;
    unsigned int index;

    if (reader->size - reader->position < 4U) {
        return false;
    }
    for (index = 0U; index < 4U; ++index) {
        int value = m3_json_hex_value(reader->data[reader->position + index]);

        if (value < 0) {
            return false;
        }
        result = (result << 4U) | (uint32_t)value;
    }
    reader->position += 4U;
    *code_point = result;
    return true;
}

static size_t m3_json_encode_utf8(uint32_t code_point, uint8_t *output)
{
    if (code_point <= 0x7fU) {
        output[0] = (uint8_t)code_point;
        return 1U;
    }
    if (code_point <= 0x7ffU) {
        output[0] = (uint8_t)(0xc0U | (code_point >> 6U));
        output[1] = (uint8_t)(0x80U | (code_point & 0x3fU));
        return 2U;
    }
    if (code_point <= 0xffffU) {
        output[0] = (uint8_t)(0xe0U | (code_point >> 12U));
        output[1] = (uint8_t)(0x80U | ((code_point >> 6U) & 0x3fU));
        output[2] = (uint8_t)(0x80U | (code_point & 0x3fU));
        return 3U;
    }
    output[0] = (uint8_t)(0xf0U | (code_point >> 18U));
    output[1] = (uint8_t)(0x80U | ((code_point >> 12U) & 0x3fU));
    output[2] = (uint8_t)(0x80U | ((code_point >> 6U) & 0x3fU));
    output[3] = (uint8_t)(0x80U | (code_point & 0x3fU));
    return 4U;
}

static size_t m3_json_utf8_length(const uint8_t *data, size_t remaining)
{
    uint8_t first;
    uint32_t code_point;
    size_t length;
    size_t index;

    if (remaining == 0U) {
        return 0U;
    }
    first = data[0];
    if (first < 0x80U) {
        return 1U;
    }
    if (first >= 0xc2U && first <= 0xdfU) {
        length = 2U;
        code_point = (uint32_t)(first & 0x1fU);
    } else if (first >= 0xe0U && first <= 0xefU) {
        length = 3U;
        code_point = (uint32_t)(first & 0x0fU);
    } else if (first >= 0xf0U && first <= 0xf4U) {
        length = 4U;
        code_point = (uint32_t)(first & 0x07U);
    } else {
        return 0U;
    }
    if (remaining < length) {
        return 0U;
    }
    for (index = 1U; index < length; ++index) {
        if ((data[index] & 0xc0U) != 0x80U) {
            return 0U;
        }
        code_point = (code_point << 6U) | (uint32_t)(data[index] & 0x3fU);
    }
    if ((length == 3U && code_point < 0x800U) ||
        (length == 4U && code_point < 0x10000U) ||
        code_point > 0x10ffffU ||
        (code_point >= 0xd800U && code_point <= 0xdfffU)) {
        return 0U;
    }
    return length;
}

static m3_status m3_json_decode_escape(m3_json_reader *reader,
                                       uint8_t *output, size_t *output_size,
                                       m3_error *error)
{
    uint8_t escape;
    uint32_t code_point;

    if (reader->position >= reader->size) {
        return m3_json_invalid(reader, error, "unterminated escape");
    }
    escape = reader->data[reader->position++];
    switch (escape) {
    case (uint8_t)'"':
    case (uint8_t)'\\':
    case (uint8_t)'/':
        output[(*output_size)++] = escape;
        return M3_STATUS_OK;
    case (uint8_t)'b':
        output[(*output_size)++] = (uint8_t)'\b';
        return M3_STATUS_OK;
    case (uint8_t)'f':
        output[(*output_size)++] = (uint8_t)'\f';
        return M3_STATUS_OK;
    case (uint8_t)'n':
        output[(*output_size)++] = (uint8_t)'\n';
        return M3_STATUS_OK;
    case (uint8_t)'r':
        output[(*output_size)++] = (uint8_t)'\r';
        return M3_STATUS_OK;
    case (uint8_t)'t':
        output[(*output_size)++] = (uint8_t)'\t';
        return M3_STATUS_OK;
    case (uint8_t)'u':
        break;
    default:
        return m3_json_invalid(reader, error, "invalid escape");
    }

    if (!m3_json_read_hex_quad(reader, &code_point)) {
        return m3_json_invalid(reader, error, "invalid Unicode escape");
    }
    if (code_point == 0U) {
        return m3_json_invalid(reader, error,
                               "NUL is not supported in JSON strings");
    }
    if (code_point >= 0xd800U && code_point <= 0xdbffU) {
        uint32_t low_surrogate;

        if (reader->size - reader->position < 6U ||
            reader->data[reader->position] != (uint8_t)'\\' ||
            reader->data[reader->position + 1U] != (uint8_t)'u') {
            return m3_json_invalid(reader, error,
                                   "missing low Unicode surrogate");
        }
        reader->position += 2U;
        if (!m3_json_read_hex_quad(reader, &low_surrogate) ||
            low_surrogate < 0xdc00U || low_surrogate > 0xdfffU) {
            return m3_json_invalid(reader, error,
                                   "invalid low Unicode surrogate");
        }
        code_point = 0x10000U + ((code_point - 0xd800U) << 10U) +
                     (low_surrogate - 0xdc00U);
    } else if (code_point >= 0xdc00U && code_point <= 0xdfffU) {
        return m3_json_invalid(reader, error,
                               "unexpected low Unicode surrogate");
    }
    *output_size += m3_json_encode_utf8(code_point, output + *output_size);
    return M3_STATUS_OK;
}

m3_status m3_json_read_string(m3_json_reader *reader, char **value,
                              m3_error *error)
{
    uint8_t *output;
    size_t output_size = 0U;
    m3_status status;

    if (value == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "JSON string output is null");
    }
    *value = NULL;
    status = m3_json_expect(reader, (uint8_t)'"', error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (reader->size - reader->position == SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "JSON string allocation overflows");
    }
    output = malloc(reader->size - reader->position + 1U);
    if (output == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate JSON string");
    }

    while (reader->position < reader->size) {
        uint8_t byte = reader->data[reader->position++];

        if (byte == (uint8_t)'"') {
            uint8_t *exact_output;

            output[output_size] = 0U;
            exact_output = realloc(output, output_size + 1U);
            if (exact_output == NULL) {
                free(output);
                return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                    "cannot retain decoded JSON string");
            }
            *value = (char *)exact_output;
            return M3_STATUS_OK;
        }
        if (byte < 0x20U) {
            free(output);
            return m3_json_invalid(reader, error,
                                   "control byte in string");
        }
        if (byte == (uint8_t)'\\') {
            status = m3_json_decode_escape(reader, output, &output_size,
                                           error);

            if (status != M3_STATUS_OK) {
                free(output);
                return status;
            }
        } else if (byte < 0x80U) {
            output[output_size++] = byte;
        } else {
            size_t sequence_position = reader->position - 1U;
            size_t sequence_length = m3_json_utf8_length(
                reader->data + sequence_position,
                reader->size - sequence_position);

            if (sequence_length == 0U) {
                free(output);
                return m3_json_invalid(reader, error,
                                       "invalid UTF-8 in string");
            }
            while (sequence_length != 0U) {
                output[output_size++] = reader->data[sequence_position++];
                sequence_length -= 1U;
            }
            reader->position = sequence_position;
        }
    }
    free(output);
    return m3_json_invalid(reader, error, "unterminated string");
}

m3_status m3_json_read_uint64(m3_json_reader *reader, uint64_t *value,
                              m3_error *error)
{
    uint64_t result = 0U;
    size_t start;

    if (reader == NULL || value == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "JSON unsigned integer argument is null");
    }
    m3_json_skip_whitespace(reader);
    start = reader->position;
    if (start >= reader->size || reader->data[start] < (uint8_t)'0' ||
        reader->data[start] > (uint8_t)'9') {
        return m3_json_invalid(reader, error,
                               "expected unsigned integer");
    }
    if (reader->data[start] == (uint8_t)'0' && start + 1U < reader->size &&
        reader->data[start + 1U] >= (uint8_t)'0' &&
        reader->data[start + 1U] <= (uint8_t)'9') {
        return m3_json_invalid(reader, error,
                               "leading zero in unsigned integer");
    }
    while (reader->position < reader->size &&
           reader->data[reader->position] >= (uint8_t)'0' &&
           reader->data[reader->position] <= (uint8_t)'9') {
        uint64_t digit =
            (uint64_t)(reader->data[reader->position] - (uint8_t)'0');

        if (result > (UINT64_MAX - digit) / 10U) {
            return m3_json_invalid(reader, error,
                                   "unsigned integer overflows uint64");
        }
        result = result * 10U + digit;
        reader->position += 1U;
    }
    *value = result;
    return M3_STATUS_OK;
}

static bool m3_json_consume_literal(m3_json_reader *reader,
                                    const char *literal, size_t length)
{
    size_t index;

    if (reader->size - reader->position < length) {
        return false;
    }
    for (index = 0U; index < length; ++index) {
        if (reader->data[reader->position + index] !=
            (uint8_t)literal[index]) {
            return false;
        }
    }
    reader->position += length;
    return true;
}

static m3_status m3_json_skip_number(m3_json_reader *reader,
                                     m3_error *error)
{
    size_t start = reader->position;

    if (reader->data[reader->position] == (uint8_t)'-') {
        reader->position += 1U;
        if (reader->position >= reader->size) {
            return m3_json_invalid(reader, error, "incomplete number");
        }
    }
    if (reader->data[reader->position] == (uint8_t)'0') {
        reader->position += 1U;
        if (reader->position < reader->size &&
            reader->data[reader->position] >= (uint8_t)'0' &&
            reader->data[reader->position] <= (uint8_t)'9') {
            return m3_json_invalid(reader, error, "leading zero in number");
        }
    } else {
        size_t digit_start = reader->position;

        while (reader->position < reader->size &&
               reader->data[reader->position] >= (uint8_t)'0' &&
               reader->data[reader->position] <= (uint8_t)'9') {
            reader->position += 1U;
        }
        if (reader->position == digit_start) {
            return m3_json_invalid(reader, error, "invalid number");
        }
    }
    if (reader->position < reader->size &&
        reader->data[reader->position] == (uint8_t)'.') {
        size_t fraction_start;

        reader->position += 1U;
        fraction_start = reader->position;
        while (reader->position < reader->size &&
               reader->data[reader->position] >= (uint8_t)'0' &&
               reader->data[reader->position] <= (uint8_t)'9') {
            reader->position += 1U;
        }
        if (reader->position == fraction_start) {
            return m3_json_invalid(reader, error, "empty number fraction");
        }
    }
    if (reader->position < reader->size &&
        (reader->data[reader->position] == (uint8_t)'e' ||
         reader->data[reader->position] == (uint8_t)'E')) {
        size_t exponent_start;

        reader->position += 1U;
        if (reader->position < reader->size &&
            (reader->data[reader->position] == (uint8_t)'+' ||
             reader->data[reader->position] == (uint8_t)'-')) {
            reader->position += 1U;
        }
        exponent_start = reader->position;
        while (reader->position < reader->size &&
               reader->data[reader->position] >= (uint8_t)'0' &&
               reader->data[reader->position] <= (uint8_t)'9') {
            reader->position += 1U;
        }
        if (reader->position == exponent_start) {
            return m3_json_invalid(reader, error, "empty number exponent");
        }
    }
    if (reader->position == start) {
        return m3_json_invalid(reader, error, "invalid number");
    }
    return M3_STATUS_OK;
}

static m3_status m3_json_skip_array(m3_json_reader *reader, m3_error *error)
{
    bool first = true;

    if (m3_json_expect(reader, (uint8_t)'[', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(reader, (uint8_t)']')) {
        m3_status status;

        if (!first &&
            m3_json_expect(reader, (uint8_t)',', error) != M3_STATUS_OK) {
            return M3_STATUS_INVALID_FORMAT;
        }
        status = m3_json_skip_value(reader, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        first = false;
    }
    return m3_json_expect(reader, (uint8_t)']', error);
}

static m3_status m3_json_skip_object(m3_json_reader *reader, m3_error *error)
{
    bool first = true;

    if (m3_json_expect(reader, (uint8_t)'{', error) != M3_STATUS_OK) {
        return M3_STATUS_INVALID_FORMAT;
    }
    while (!m3_json_next_is(reader, (uint8_t)'}')) {
        char *key = NULL;
        m3_status status;

        if (!first &&
            m3_json_expect(reader, (uint8_t)',', error) != M3_STATUS_OK) {
            return M3_STATUS_INVALID_FORMAT;
        }
        status = m3_json_read_string(reader, &key, error);
        free(key);
        if (status != M3_STATUS_OK) {
            return status;
        }
        if (m3_json_expect(reader, (uint8_t)':', error) != M3_STATUS_OK) {
            return M3_STATUS_INVALID_FORMAT;
        }
        status = m3_json_skip_value(reader, error);
        if (status != M3_STATUS_OK) {
            return status;
        }
        first = false;
    }
    return m3_json_expect(reader, (uint8_t)'}', error);
}

m3_status m3_json_skip_value(m3_json_reader *reader, m3_error *error)
{
    m3_status status;
    uint8_t byte;

    if (reader == NULL || reader->data == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "JSON reader is null");
    }
    m3_json_skip_whitespace(reader);
    if (reader->position >= reader->size) {
        return m3_json_invalid(reader, error, "missing value");
    }
    if (reader->depth >= M3_JSON_MAX_DEPTH) {
        return m3_json_invalid(reader, error, "nesting limit exceeded");
    }
    reader->depth += 1U;
    byte = reader->data[reader->position];
    if (byte == (uint8_t)'{') {
        status = m3_json_skip_object(reader, error);
    } else if (byte == (uint8_t)'[') {
        status = m3_json_skip_array(reader, error);
    } else if (byte == (uint8_t)'"') {
        char *value = NULL;

        status = m3_json_read_string(reader, &value, error);
        free(value);
    } else if (byte == (uint8_t)'t') {
        status = m3_json_consume_literal(reader, "true", 4U)
                     ? M3_STATUS_OK
                     : m3_json_invalid(reader, error, "invalid literal");
    } else if (byte == (uint8_t)'f') {
        status = m3_json_consume_literal(reader, "false", 5U)
                     ? M3_STATUS_OK
                     : m3_json_invalid(reader, error, "invalid literal");
    } else if (byte == (uint8_t)'n') {
        status = m3_json_consume_literal(reader, "null", 4U)
                     ? M3_STATUS_OK
                     : m3_json_invalid(reader, error, "invalid literal");
    } else if (byte == (uint8_t)'-' ||
               (byte >= (uint8_t)'0' && byte <= (uint8_t)'9')) {
        status = m3_json_skip_number(reader, error);
    } else {
        status = m3_json_invalid(reader, error, "invalid value");
    }
    reader->depth -= 1U;
    return status;
}

m3_status m3_json_finish(m3_json_reader *reader, m3_error *error)
{
    if (reader == NULL || reader->data == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "JSON reader is null");
    }
    m3_json_skip_whitespace(reader);
    if (reader->position != reader->size) {
        return m3_json_invalid(reader, error,
                               "trailing bytes after document");
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}

m3_status m3_json_validate(const uint8_t *data, size_t size, m3_error *error)
{
    m3_json_reader reader;
    m3_status status;

    if (data == NULL || size == 0U) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "JSON document is empty");
    }
    m3_json_reader_init(&reader, data, size);
    status = m3_json_skip_value(&reader, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    return m3_json_finish(&reader, error);
}
