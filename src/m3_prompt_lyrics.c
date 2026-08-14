/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_prompt_internal.h"

#include <stdbool.h>

static bool m3_leading_tag_run(const char *line, size_t length,
                               size_t *first, size_t *last)
{
    size_t cursor = 0U;
    size_t end;
    size_t next;
    bool found = false;

    while (cursor < length && (line[cursor] == ' ' || line[cursor] == '\t')) {
        ++cursor;
    }
    *first = cursor;
    while (cursor < length && line[cursor] == '[') {
        end = cursor + 1U;
        while (end < length && line[end] != ']') {
            ++end;
        }
        if (end == length || end == cursor + 1U) {
            break;
        }
        found = true;
        cursor = end + 1U;
        *last = cursor;
        next = cursor;
        while (next < length &&
               (line[next] == ' ' || line[next] == '\t')) {
            ++next;
        }
        if (next >= length || line[next] != '[') {
            break;
        }
        cursor = next;
    }
    if (!found) {
        *last = *first;
    }
    return found;
}

static m3_status m3_filter_lyrics_lines(const m3_prompt_text *input,
                                        m3_prompt_text *output,
                                        m3_error *error)
{
    m3_text_builder builder = {0};
    size_t cursor = 0U;
    bool wrote_line = false;
    m3_status status = M3_STATUS_OK;

    m3_prompt_text_init(output);
    while (cursor <= input->length) {
        size_t end = cursor;
        size_t first;
        size_t last;
        const char *line = input->data + cursor;
        size_t length;

        while (end < input->length && input->data[end] != '\n') {
            ++end;
        }
        length = end - cursor;
        if (m3_leading_tag_run(line, length, &first, &last)) {
            line += first;
            length = last - first;
        }
        if (wrote_line) {
            status = m3_builder_byte(&builder, '\n', error);
        }
        if (status == M3_STATUS_OK) {
            status = m3_builder_append(&builder, line, length, error);
        }
        if (status != M3_STATUS_OK || end == input->length) {
            break;
        }
        wrote_line = true;
        cursor = end + 1U;
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_finish(&builder, output, error);
    }
    if (status != M3_STATUS_OK) {
        m3_builder_dispose(&builder);
    }
    return status;
}

static m3_status m3_replace_pair(const m3_prompt_text *input, char first,
                                 char second, const char replacement[2],
                                 m3_prompt_text *output, m3_error *error)
{
    m3_text_builder builder = {0};
    size_t cursor = 0U;
    m3_status status = M3_STATUS_OK;

    m3_prompt_text_init(output);
    while (cursor < input->length) {
        if (cursor + 1U < input->length && input->data[cursor] == first &&
            input->data[cursor + 1U] == second) {
            status = m3_builder_append(&builder, replacement, 2U, error);
            cursor += 2U;
        } else {
            status = m3_builder_byte(&builder, input->data[cursor], error);
            ++cursor;
        }
        if (status != M3_STATUS_OK) {
            break;
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_finish(&builder, output, error);
    }
    if (status != M3_STATUS_OK) {
        m3_builder_dispose(&builder);
    }
    return status;
}

static m3_status m3_replace_caret(const m3_prompt_text *input,
                                  m3_prompt_text *output, m3_error *error)
{
    m3_text_builder builder = {0};
    size_t cursor = 0U;
    m3_status status = M3_STATUS_OK;

    m3_prompt_text_init(output);
    while (cursor < input->length) {
        if (cursor + 2U < input->length && input->data[cursor] == ' ' &&
            input->data[cursor + 1U] == '^' &&
            input->data[cursor + 2U] == ' ') {
            status = m3_builder_byte(&builder, '\n', error);
            cursor += 3U;
        } else {
            status = m3_builder_byte(&builder, input->data[cursor], error);
            ++cursor;
        }
        if (status != M3_STATUS_OK) {
            break;
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_finish(&builder, output, error);
    }
    if (status != M3_STATUS_OK) {
        m3_builder_dispose(&builder);
    }
    return status;
}

static void m3_lowercase_tags(m3_prompt_text *lyrics)
{
    size_t cursor = 0U;

    while (cursor < lyrics->length) {
        size_t close;
        size_t index;

        if (lyrics->data[cursor] != '[') {
            ++cursor;
            continue;
        }
        close = cursor + 1U;
        while (close < lyrics->length && lyrics->data[close] != ']') {
            ++close;
        }
        if (close == lyrics->length) {
            break;
        }
        for (index = cursor + 1U; index < close; ++index) {
            if (lyrics->data[index] >= 'A' && lyrics->data[index] <= 'Z') {
                lyrics->data[index] =
                    (char)(lyrics->data[index] - 'A' + 'a');
            }
        }
        cursor = close + 1U;
    }
}

m3_status m3_prompt_clean_lyrics(const m3_prompt_text *input,
                                 m3_prompt_text *output, m3_error *error)
{
    m3_prompt_text filtered;
    m3_prompt_text close_space;
    m3_prompt_text open_space;
    m3_prompt_text caret;
    m3_text_builder builder = {0};
    m3_status status;

    m3_prompt_text_init(&filtered);
    m3_prompt_text_init(&close_space);
    m3_prompt_text_init(&open_space);
    m3_prompt_text_init(&caret);
    m3_prompt_text_init(output);
    status = m3_filter_lyrics_lines(input, &filtered, error);
    if (status == M3_STATUS_OK) {
        status = m3_replace_pair(&filtered, ']', ' ', "]\n", &close_space,
                                 error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_replace_pair(&close_space, ' ', '[', "\n[", &open_space,
                                 error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_replace_caret(&open_space, &caret, error);
    }
    if (status == M3_STATUS_OK) {
        m3_lowercase_tags(&caret);
        status = m3_builder_append(&builder, "[start]\n", 8U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_append(&builder, caret.data, caret.length, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_finish(&builder, output, error);
    }
    m3_prompt_text_dispose(&filtered);
    m3_prompt_text_dispose(&close_space);
    m3_prompt_text_dispose(&open_space);
    m3_prompt_text_dispose(&caret);
    if (status != M3_STATUS_OK) {
        m3_builder_dispose(&builder);
    }
    return status;
}
