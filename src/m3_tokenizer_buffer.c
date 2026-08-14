/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_tokenizer_internal.h"

#include <stdlib.h>
#include <string.h>

m3_status m3_byte_buffer_reserve(m3_byte_buffer *buffer, size_t extra,
                                  m3_error *error)
{
    size_t required;
    size_t capacity;
    uint8_t *data;

    if (buffer == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "byte buffer is null");
    }
    if (extra > SIZE_MAX - buffer->count) {
        return m3_error_set(error, M3_STATUS_OVERFLOW,
                            "byte buffer size overflows");
    }
    required = buffer->count + extra;
    if (required <= buffer->capacity) {
        return M3_STATUS_OK;
    }
    capacity = buffer->capacity == 0U ? 256U : buffer->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2U) {
            capacity = required;
            break;
        }
        capacity *= 2U;
    }
    data = realloc(buffer->data, capacity);
    if (data == NULL) {
        return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                            "cannot grow byte buffer to %zu bytes", capacity);
    }
    buffer->data = data;
    buffer->capacity = capacity;
    return M3_STATUS_OK;
}

m3_status m3_byte_buffer_append(m3_byte_buffer *buffer, const void *data,
                                 size_t size, m3_error *error)
{
    m3_status status;

    if (size != 0U && data == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "byte buffer source is null");
    }
    status = m3_byte_buffer_reserve(buffer, size, error);
    if (status != M3_STATUS_OK) {
        return status;
    }
    if (size != 0U) {
        (void)memcpy(buffer->data + buffer->count, data, size);
        buffer->count += size;
    }
    return M3_STATUS_OK;
}

void m3_byte_buffer_dispose(m3_byte_buffer *buffer)
{
    if (buffer != NULL) {
        free(buffer->data);
        buffer->data = NULL;
        buffer->count = 0U;
        buffer->capacity = 0U;
    }
}

bool m3_tokenizer_string_equal(const m3_byte_buffer *buffer,
                               m3_token_span span, const char *literal)
{
    size_t length;

    if (buffer == NULL || literal == NULL ||
        (size_t)span.offset > buffer->count ||
        (size_t)span.length > buffer->count - (size_t)span.offset) {
        return false;
    }
    length = strlen(literal);
    if (length != (size_t)span.length) {
        return false;
    }
    if (length == 0U) {
        return true;
    }
    return buffer->data != NULL &&
           (size_t)span.offset <= buffer->count &&
           (size_t)span.length <= buffer->count - (size_t)span.offset &&
           memcmp(buffer->data + span.offset, literal, length) == 0;
}
