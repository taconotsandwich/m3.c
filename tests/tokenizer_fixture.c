/* SPDX-License-Identifier: GPL-2.0-only */

#include "tokenizer_fixture.h"

#include "m3_unicode.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t *data;
    size_t length;
} m3_fixture_name;

static bool m3_fixture_reserve(m3_byte_buffer *buffer, size_t extra)
{
    size_t required;
    size_t capacity;
    uint8_t *data;

    if (extra > SIZE_MAX - buffer->count) {
        return false;
    }
    required = buffer->count + extra;
    if (required <= buffer->capacity) {
        return true;
    }
    capacity = buffer->capacity == 0U ? 4096U : buffer->capacity;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2U) {
            capacity = required;
            break;
        }
        capacity *= 2U;
    }
    data = realloc(buffer->data, capacity);
    if (data == NULL) {
        return false;
    }
    buffer->data = data;
    buffer->capacity = capacity;
    return true;
}

static bool m3_fixture_append(m3_byte_buffer *buffer, const void *data,
                              size_t size)
{
    if (!m3_fixture_reserve(buffer, size)) {
        return false;
    }
    if (size != 0U) {
        (void)memcpy(buffer->data + buffer->count, data, size);
        buffer->count += size;
    }
    return true;
}

static bool m3_fixture_text(m3_byte_buffer *buffer, const char *text)
{
    return m3_fixture_append(buffer, text, strlen(text));
}

static bool m3_fixture_format(m3_byte_buffer *buffer, const char *format, ...)
{
    char text[128];
    va_list arguments;
    int length;

    va_start(arguments, format);
    length = vsnprintf(text, sizeof(text), format, arguments);
    va_end(arguments);
    return length >= 0 && (size_t)length < sizeof(text) &&
           m3_fixture_append(buffer, text, (size_t)length);
}

static bool m3_fixture_json_string(m3_byte_buffer *buffer,
                                   const uint8_t *data, size_t size)
{
    size_t index;

    if (!m3_fixture_text(buffer, "\"")) {
        return false;
    }
    for (index = 0U; index < size; ++index) {
        if ((data[index] == (uint8_t)'\"' ||
             data[index] == (uint8_t)'\\') &&
            !m3_fixture_text(buffer, "\\")) {
            return false;
        }
        if (!m3_fixture_append(buffer, data + index, 1U)) {
            return false;
        }
    }
    return m3_fixture_text(buffer, "\"");
}

static void m3_fixture_names_dispose(m3_fixture_name *names, size_t count)
{
    size_t index;

    if (names != NULL) {
        for (index = 0U; index < count; ++index) {
            free(names[index].data);
        }
        free(names);
    }
}

static m3_fixture_name *m3_fixture_names(const m3_fixture_merge *merges,
                                          size_t merge_count)
{
    size_t count = 256U + merge_count;
    m3_fixture_name *names = calloc(count, sizeof(*names));
    size_t index;

    if (names == NULL) {
        return NULL;
    }
    for (index = 0U; index < 256U; ++index) {
        uint8_t encoded[4];
        size_t length = m3_utf8_encode(
            m3_byte_alphabet_code_point((uint8_t)index), encoded);

        names[index].data = malloc(length);
        if (names[index].data == NULL) {
            m3_fixture_names_dispose(names, count);
            return NULL;
        }
        (void)memcpy(names[index].data, encoded, length);
        names[index].length = length;
    }
    for (index = 0U; index < merge_count; ++index) {
        uint32_t left = merges[index].left;
        uint32_t right = merges[index].right;
        size_t output = 256U + index;
        size_t length;

        if (left >= output || right >= output ||
            names[left].length > SIZE_MAX - names[right].length) {
            m3_fixture_names_dispose(names, count);
            return NULL;
        }
        length = names[left].length + names[right].length;
        names[output].data = malloc(length);
        if (names[output].data == NULL) {
            m3_fixture_names_dispose(names, count);
            return NULL;
        }
        (void)memcpy(names[output].data, names[left].data,
                     names[left].length);
        (void)memcpy(names[output].data + names[left].length,
                     names[right].data, names[right].length);
        names[output].length = length;
    }
    return names;
}

static const char *m3_fixture_added_content(uint32_t id, char buffer[64],
                                            m3_tokenizer_fixture_kind kind)
{
    switch (id) {
    case M3_TOKEN_END_OF_TEXT: return "<|endoftext|>";
    case M3_TOKEN_IM_START: return "<|im_start|>";
    case M3_TOKEN_IM_END: return "<|im_end|>";
    case M3_TOKEN_AUDIO_CFG:
        return kind == M3_TOKENIZER_FIXTURE_MISSING_MARKER
                   ? "<|audio_bad|>"
                   : "<|audio_cfg|>";
    case M3_TOKEN_AUDIO_START: return "<|audio_start|>";
    case M3_TOKEN_AUDIO_END: return "<|audio_end|>";
    case M3_TOKEN_CAPTION_START: return "<|caption_start|>";
    case M3_TOKEN_CAPTION_END: return "<|caption_end|>";
    case M3_TOKEN_LYRICS_START: return "<|lyrics_start|>";
    case M3_TOKEN_LYRICS_END: return "<|lyrics_end|>";
    default:
        (void)snprintf(buffer, 64U,
                       id <= 151656U ? "<|special_%u|>" : "<ordinary_%u>",
                       id);
        return buffer;
    }
}

static bool m3_fixture_added(m3_byte_buffer *json,
                             m3_tokenizer_fixture_kind kind)
{
    uint32_t id;

    if (!m3_fixture_text(json, "\"added_tokens\":[")) {
        return false;
    }
    for (id = M3_TOKEN_END_OF_TEXT; id < M3_TOKENIZER_ID_COUNT; ++id) {
        char content_buffer[64];
        const char *content = m3_fixture_added_content(id, content_buffer,
                                                       kind);
        uint32_t written_id = id;
        bool special = (id <= 151656U || id >= 151669U);

        if (kind == M3_TOKENIZER_FIXTURE_WRONG_ADDED_ID &&
            id == M3_TOKEN_AUDIO_CFG) {
            written_id = M3_TOKEN_AUDIO_CFG + 1U;
        }
        if (id != M3_TOKEN_END_OF_TEXT && !m3_fixture_text(json, ",")) {
            return false;
        }
        if (!m3_fixture_format(json, "{\"id\":%u,\"content\":", written_id) ||
            !m3_fixture_json_string(json, (const uint8_t *)content,
                                    strlen(content)) ||
            !m3_fixture_format(
                json,
                ",\"single_word\":false,\"lstrip\":false,"
                "\"rstrip\":false,\"normalized\":false,"
                "\"special\":%s}", special ? "true" : "false")) {
            return false;
        }
    }
    return m3_fixture_text(json, "],");
}

static bool m3_fixture_metadata(m3_byte_buffer *json,
                                m3_tokenizer_fixture_kind kind)
{
    static const char text[] =
        "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,";
    static const char normalizer_prefix[] =
        "\"normalizer\":{\"type\":\"NFC\"},"
        "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
        "{\"type\":\"Split\",\"pattern\":";
    static const char pattern[] =
        "{\"Regex\":"
        "\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?"
        "\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+"
        "[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"}";
    static const char normalizer_suffix[] =
        ","
        "\"behavior\":\"Isolated\",\"invert\":false},"
        "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
        "\"trim_offsets\":true,\"use_regex\":false}]},"
        "\"post_processor\":{\"type\":\"ByteLevel\","
        "\"add_prefix_space\":false,\"trim_offsets\":false,"
        "\"use_regex\":false},"
        "\"decoder\":{\"type\":\"ByteLevel\","
        "\"add_prefix_space\":true,\"trim_offsets\":true,"
        "\"use_regex\":true},";

    return m3_fixture_text(json, text) &&
           m3_fixture_text(json, normalizer_prefix) &&
           m3_fixture_text(json,
                           kind == M3_TOKENIZER_FIXTURE_EMPTY_PATTERN
                               ? "{}"
                               : pattern) &&
           m3_fixture_text(json, normalizer_suffix);
}

static bool m3_fixture_model(m3_byte_buffer *json,
                              const m3_fixture_name *names,
                              const m3_fixture_merge *merges,
                              size_t merge_count,
                              m3_tokenizer_fixture_kind kind)
{
    size_t vocab_count = 256U + merge_count;
    size_t index;

    if (!m3_fixture_text(
            json,
            kind == M3_TOKENIZER_FIXTURE_EMPTY_MODEL_FIELD
                ? "\"model\":{\"\":null,\"type\":\"BPE\",\"dropout\":null,"
                : "\"model\":{\"type\":\"BPE\",\"dropout\":null,") ||
        !m3_fixture_text(
            json,
            "\"unk_token\":null,\"continuing_subword_prefix\":\"\","
            "\"end_of_word_suffix\":\"\",\"fuse_unk\":false,"
            "\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":{")) {
        return false;
    }
    for (index = 0U; index < vocab_count; ++index) {
        if (index != 0U && !m3_fixture_text(json, ",")) {
            return false;
        }
        if (!m3_fixture_json_string(
                json, names[index].data,
                kind == M3_TOKENIZER_FIXTURE_EMPTY_VOCAB_TOKEN && index == 0U
                    ? 0U
                    : names[index].length) ||
            !m3_fixture_format(json, ":%zu", index)) {
            return false;
        }
    }
    if (kind == M3_TOKENIZER_FIXTURE_DUPLICATE_VOCAB &&
        (!m3_fixture_text(json, ",") ||
         !m3_fixture_json_string(json, names[0].data, names[0].length) ||
         !m3_fixture_text(json, ":0"))) {
        return false;
    }
    if (!m3_fixture_text(json, "},\"merges\":[")) {
        return false;
    }
    for (index = 0U; index < merge_count; ++index) {
        size_t source = kind == M3_TOKENIZER_FIXTURE_DUPLICATE_MERGE &&
                                index == 1U
                            ? 0U
                            : index;
        uint32_t left = merges[source].left;
        uint32_t right = merges[source].right;

        if (index != 0U && !m3_fixture_text(json, ",")) {
            return false;
        }
        if (!m3_fixture_text(json, "[") ||
            !m3_fixture_json_string(
                json, names[left].data,
                kind == M3_TOKENIZER_FIXTURE_EMPTY_MERGE_TOKEN && index == 0U
                    ? 0U
                    : names[left].length) ||
            !m3_fixture_text(json, ",") ||
            !m3_fixture_json_string(json, names[right].data,
                                    names[right].length) ||
            !m3_fixture_text(json, "]")) {
            return false;
        }
    }
    return m3_fixture_text(json, "]}}");
}

bool m3_tokenizer_fixture_build(m3_tokenizer_fixture *fixture,
                                const m3_fixture_merge *merges,
                                size_t merge_count,
                                m3_tokenizer_fixture_kind kind)
{
    m3_byte_buffer json = {0};
    m3_fixture_name *names;
    bool success;

    if (fixture == NULL || (merge_count != 0U && merges == NULL) ||
        merge_count > UINT32_MAX - 256U) {
        return false;
    }
    fixture->data = NULL;
    fixture->size = 0U;
    fixture->contract.model_vocab_count = (uint32_t)(256U + merge_count);
    fixture->contract.merge_count = (uint32_t)merge_count;
    names = m3_fixture_names(merges, merge_count);
    if (names == NULL) {
        return false;
    }
    success = m3_fixture_metadata(&json, kind) &&
              m3_fixture_added(&json, kind) &&
              m3_fixture_model(&json, names, merges, merge_count, kind);
    m3_fixture_names_dispose(names, 256U + merge_count);
    if (!success) {
        free(json.data);
        return false;
    }
    fixture->data = json.data;
    fixture->size = json.count;
    return true;
}

void m3_tokenizer_fixture_dispose(m3_tokenizer_fixture *fixture)
{
    if (fixture != NULL) {
        free(fixture->data);
        fixture->data = NULL;
        fixture->size = 0U;
        fixture->contract.model_vocab_count = 0U;
        fixture->contract.merge_count = 0U;
    }
}
