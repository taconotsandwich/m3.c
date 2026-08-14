/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_prompt_internal.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static m3_status m3_expand_caption_fields(const m3_prompt_text *input,
                                          m3_prompt_text *output,
                                          m3_error *error)
{
    m3_text_builder builder = {0};
    size_t cursor = 0U;
    m3_status status = M3_STATUS_OK;

    m3_prompt_text_init(output);
    while (cursor < input->length) {
        size_t open = m3_find_bytes(input->data, input->length, cursor,
                                    "<|", 2U);
        size_t close;
        size_t first;
        size_t last;
        size_t split;

        if (open == SIZE_MAX) {
            status = m3_builder_append(&builder, input->data + cursor,
                                       input->length - cursor, error);
            break;
        }
        status = m3_builder_append(&builder, input->data + cursor,
                                   open - cursor, error);
        if (status != M3_STATUS_OK) {
            break;
        }
        close = m3_find_bytes(input->data, input->length, open + 2U,
                              "|>", 2U);
        if (close == SIZE_MAX) {
            status = m3_builder_append(&builder, input->data + open,
                                       input->length - open, error);
            break;
        }
        if (memchr(input->data + open + 2U, '|', close - open - 2U) != NULL) {
            status = m3_builder_append(&builder, input->data + open, 2U,
                                       error);
            if (status != M3_STATUS_OK) {
                break;
            }
            cursor = open + 2U;
            continue;
        }
        first = open + 2U;
        last = close;
        while (first < last && m3_ascii_space(input->data[first])) {
            ++first;
        }
        while (last > first && m3_ascii_space(input->data[last - 1U])) {
            --last;
        }
        split = first;
        while (split < last && !m3_ascii_space(input->data[split])) {
            ++split;
        }
        status = m3_builder_append(&builder, input->data + first,
                                   split - first, error);
        while (split < last && m3_ascii_space(input->data[split])) {
            ++split;
        }
        if (status == M3_STATUS_OK && split < last) {
            status = m3_builder_append(&builder, " is ", 4U, error);
        }
        if (status == M3_STATUS_OK && split < last) {
            status = m3_builder_append(&builder, input->data + split,
                                       last - split, error);
        }
        if (status != M3_STATUS_OK) {
            break;
        }
        cursor = close + 2U;
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_finish(&builder, output, error);
    }
    if (status != M3_STATUS_OK) {
        m3_builder_dispose(&builder);
    }
    return status;
}

static void m3_strip_caption_prefix(const char **line, size_t *length)
{
    size_t spaces = 0U;
    size_t cursor;
    size_t marks = 0U;

    while (spaces < *length && spaces < 4U &&
           m3_horizontal_space((*line)[spaces])) {
        ++spaces;
    }
    cursor = spaces;
    while (cursor < *length && (*line)[cursor] == '#') {
        ++cursor;
        ++marks;
    }
    if (spaces <= 3U && marks >= 1U && marks <= 6U && cursor < *length &&
        m3_horizontal_space((*line)[cursor])) {
        while (cursor < *length && m3_horizontal_space((*line)[cursor])) {
            ++cursor;
        }
        *line += cursor;
        *length -= cursor;
    }

    cursor = 0U;
    while (cursor < *length && m3_horizontal_space((*line)[cursor])) {
        ++cursor;
    }
    if (cursor + 1U < *length &&
        ((*line)[cursor] == '*' || (*line)[cursor] == '+' ||
         (*line)[cursor] == '-') &&
        m3_horizontal_space((*line)[cursor + 1U])) {
        cursor += 2U;
        while (cursor < *length && m3_horizontal_space((*line)[cursor])) {
            ++cursor;
        }
        *line += cursor;
        *length -= cursor;
    }
}

static bool m3_horizontal_rule(const char *line, size_t length)
{
    size_t first = 0U;
    size_t last = length;
    size_t index;

    while (first < last && m3_horizontal_space(line[first])) {
        ++first;
    }
    while (last > first && m3_horizontal_space(line[last - 1U])) {
        --last;
    }
    if (last - first < 3U) {
        return false;
    }
    for (index = first; index < last; ++index) {
        if (line[index] != '-' && line[index] != '*' && line[index] != '_') {
            return false;
        }
    }
    return true;
}

static void m3_remove_markup_pair(char *line, size_t *length, size_t open,
                                  size_t close, size_t delimiter_length)
{
    (void)memmove(line + close, line + close + delimiter_length,
                  *length - close - delimiter_length);
    *length -= delimiter_length;
    (void)memmove(line + open, line + open + delimiter_length,
                  *length - open - delimiter_length);
    *length -= delimiter_length;
    line[*length] = '\0';
}

static void m3_unwrap_bold(char *line, size_t *length)
{
    size_t open = 0U;

    while (open + 2U <= *length) {
        size_t close;

        if (line[open] != '*' || line[open + 1U] != '*') {
            ++open;
            continue;
        }
        close = open + 2U;
        while (close < *length && line[close] != '*') {
            ++close;
        }
        if (close == open + 2U || close + 1U >= *length ||
            line[close + 1U] != '*') {
            ++open;
            continue;
        }
        m3_remove_markup_pair(line, length, open, close, 2U);
        open = 0U;
    }
}

static void m3_unwrap_italic(char *line, size_t *length)
{
    size_t open = 0U;

    while (open < *length) {
        size_t close;

        if (line[open] != '*' || (open > 0U && line[open - 1U] == '*') ||
            open + 1U >= *length || line[open + 1U] == '*') {
            ++open;
            continue;
        }
        close = open + 1U;
        while (close < *length && line[close] != '*') {
            ++close;
        }
        if (close == open + 1U || close == *length ||
            (close + 1U < *length && line[close + 1U] == '*')) {
            ++open;
            continue;
        }
        m3_remove_markup_pair(line, length, open, close, 1U);
        open = 0U;
    }
}

static m3_status m3_clean_caption_lines(const m3_prompt_text *input,
                                        m3_prompt_text *output,
                                        m3_error *error)
{
    m3_text_builder lines = {0};
    m3_text_builder clean = {0};
    size_t cursor = 0U;
    bool wrote_line = false;
    m3_status status = M3_STATUS_OK;

    m3_prompt_text_init(output);
    while (cursor < input->length) {
        size_t end = cursor;
        const char *line;
        size_t length;
        char *copy;

        while (end < input->length && input->data[end] != '\n') {
            ++end;
        }
        line = input->data + cursor;
        length = end - cursor;
        m3_strip_caption_prefix(&line, &length);
        copy = malloc(length + 1U);
        if (copy == NULL) {
            status = m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                  "cannot allocate caption line");
            break;
        }
        (void)memcpy(copy, line, length);
        copy[length] = '\0';
        m3_unwrap_bold(copy, &length);
        m3_unwrap_italic(copy, &length);
        while (length > 0U && m3_horizontal_space(copy[length - 1U])) {
            --length;
        }
        copy[length] = '\0';
        if (!m3_horizontal_rule(copy, length)) {
            if (wrote_line) {
                status = m3_builder_byte(&lines, '\n', error);
            }
            if (status == M3_STATUS_OK) {
                status = m3_builder_append(&lines, copy, length, error);
            }
            wrote_line = true;
        }
        free(copy);
        if (status != M3_STATUS_OK) {
            break;
        }
        if (end == input->length) {
            break;
        }
        cursor = end + 1U;
    }

    for (cursor = 0U; status == M3_STATUS_OK && cursor < lines.length;) {
        if (cursor + 4U <= lines.length &&
            memcmp(lines.data + cursor, "\xE2\x80\xA2 ", 4U) == 0) {
            cursor += 4U;
        } else if (cursor + 4U <= lines.length &&
                   memcmp(lines.data + cursor, "    ", 4U) == 0) {
            cursor += 4U;
        } else if (lines.data[cursor] == '\n' && clean.length > 0U &&
                   clean.data[clean.length - 1U] == '\n') {
            ++cursor;
        } else {
            status = m3_builder_byte(&clean, lines.data[cursor], error);
            ++cursor;
        }
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_finish(&clean, output, error);
    }
    m3_builder_dispose(&lines);
    if (status != M3_STATUS_OK) {
        m3_builder_dispose(&clean);
    }
    return status;
}

m3_status m3_prompt_clean_caption(const m3_prompt_text *input,
                                  m3_prompt_text *output, m3_error *error)
{
    m3_prompt_text expanded;
    m3_status status;

    m3_prompt_text_init(&expanded);
    status = m3_expand_caption_fields(input, &expanded, error);
    if (status == M3_STATUS_OK) {
        status = m3_clean_caption_lines(&expanded, output, error);
    }
    m3_prompt_text_dispose(&expanded);
    return status;
}
