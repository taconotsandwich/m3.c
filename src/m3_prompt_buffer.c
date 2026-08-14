/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_prompt_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

bool m3_ascii_space(char byte)
{
    return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' ||
           byte == '\v' || byte == '\f';
}

bool m3_horizontal_space(char byte)
{
    return byte == ' ' || byte == '\t' || byte == '\v' || byte == '\f';
}

static bool m3_size_add(size_t left, size_t right, size_t *result)
{
    if (right > SIZE_MAX - left) {
        return false;
    }
    *result = left + right;
    return true;
}

void m3_builder_dispose(m3_text_builder *builder)
{
    free(builder->data);
    (void)memset(builder, 0, sizeof(*builder));
}

static m3_status m3_builder_reserve(m3_text_builder *builder, size_t added,
                                    m3_error *error)
{
    size_t required;
    size_t capacity;
    char *allocation;

    if (!m3_size_add(builder->length, added, &required) ||
        required == SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "prompt byte count overflows size_t");
    }
    ++required;
    if (required <= builder->capacity) {
        return M3_STATUS_OK;
    }

    capacity = builder->capacity == 0U ? 64U : builder->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2U) {
            capacity = required;
            break;
        }
        capacity *= 2U;
    }
    allocation = realloc(builder->data, capacity);
    if (allocation == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot allocate prompt text");
    }
    builder->data = allocation;
    builder->capacity = capacity;
    return M3_STATUS_OK;
}

m3_status m3_builder_append(m3_text_builder *builder, const char *data,
                            size_t length, m3_error *error)
{
    m3_status status;

    if (length == 0U) {
        return M3_STATUS_OK;
    }
    status = m3_builder_reserve(builder, length, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    (void)memcpy(builder->data + builder->length, data, length);
    builder->length += length;
    builder->data[builder->length] = '\0';
    return M3_STATUS_OK;
}

m3_status m3_builder_byte(m3_text_builder *builder, char byte,
                          m3_error *error)
{
    return m3_builder_append(builder, &byte, 1U, error);
}

m3_status m3_builder_finish(m3_text_builder *builder, m3_prompt_text *text,
                            m3_error *error)
{
    m3_status status = m3_builder_reserve(builder, 0U, error);

    if (status != M3_STATUS_OK) {
        return status;
    }
    builder->data[builder->length] = '\0';
    text->data = builder->data;
    text->length = builder->length;
    (void)memset(builder, 0, sizeof(*builder));
    return M3_STATUS_OK;
}

static bool m3_text_has_content(const m3_prompt_text *text)
{
    size_t index;

    for (index = 0U; index < text->length; ++index) {
        if (!m3_ascii_space(text->data[index])) {
            return true;
        }
    }
    return false;
}

m3_status m3_normalize_input(m3_text_view input, const char *name,
                             m3_prompt_text *normalized, m3_error *error)
{
    m3_text_builder builder = {0};
    size_t index;
    m3_status status;

    m3_prompt_text_init(normalized);
    if (input.data == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s input is null", name);
    }
    if (input.length == SIZE_MAX) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "%s input length overflows prompt storage", name);
    }
    if (memchr(input.data, '\0', input.length) != NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                            "%s input contains an embedded NUL byte", name);
    }

    for (index = 0U; index < input.length; ++index) {
        char byte = input.data[index];

        if (byte == '\r') {
            if (index + 1U < input.length && input.data[index + 1U] == '\n') {
                ++index;
            }
            byte = '\n';
        }
        status = m3_builder_byte(&builder, byte, error);
        if (status != M3_STATUS_OK) {
            m3_builder_dispose(&builder);
            return status;
        }
    }
    status = m3_builder_finish(&builder, normalized, error);
    if (status != M3_STATUS_OK) {
        m3_builder_dispose(&builder);
        return status;
    }
    if (!m3_text_has_content(normalized)) {
        m3_prompt_text_dispose(normalized);
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "%s input is empty after trimming", name);
    }
    return M3_STATUS_OK;
}

size_t m3_find_bytes(const char *data, size_t length, size_t start,
                     const char *needle, size_t needle_length)
{
    size_t index;

    if (needle_length == 0U || start > length ||
        needle_length > length - start) {
        return SIZE_MAX;
    }
    for (index = start; index <= length - needle_length; ++index) {
        if (memcmp(data + index, needle, needle_length) == 0) {
            return index;
        }
    }
    return SIZE_MAX;
}
