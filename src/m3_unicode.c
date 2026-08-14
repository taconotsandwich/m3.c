/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_unicode.h"

#include "m3_unicode_internal.h"

#include <stdlib.h>
#include <string.h>

#define M3_HANGUL_S_BASE 0xAC00U
#define M3_HANGUL_L_BASE 0x1100U
#define M3_HANGUL_V_BASE 0x1161U
#define M3_HANGUL_T_BASE 0x11A7U
#define M3_HANGUL_L_COUNT 19U
#define M3_HANGUL_V_COUNT 21U
#define M3_HANGUL_T_COUNT 28U
#define M3_HANGUL_N_COUNT (M3_HANGUL_V_COUNT * M3_HANGUL_T_COUNT)
#define M3_HANGUL_S_COUNT (M3_HANGUL_L_COUNT * M3_HANGUL_N_COUNT)

typedef struct {
    uint32_t *items;
    size_t count;
    size_t capacity;
} m3_code_points;

static bool m3_range_contains(const m3_unicode_range_shard *shards,
                              size_t shard_count, uint32_t code_point)
{
    size_t shard_index;

    for (shard_index = 0U; shard_index < shard_count; ++shard_index) {
        const m3_unicode_range *items = shards[shard_index].items;
        size_t count = shards[shard_index].count;
        size_t low = 0U;
        size_t high = count;

        if (count == 0U || code_point < items[0].first ||
            code_point > items[count - 1U].last) {
            continue;
        }
        while (low < high) {
            size_t middle = low + (high - low) / 2U;

            if (code_point < items[middle].first) {
                high = middle;
            } else if (code_point > items[middle].last) {
                low = middle + 1U;
            } else {
                return true;
            }
        }
    }
    return false;
}

bool m3_unicode_is_letter(uint32_t code_point)
{
    return m3_range_contains(m3_unicode_letter_shards,
                             m3_unicode_letter_shard_count, code_point);
}

bool m3_unicode_is_number(uint32_t code_point)
{
    return m3_range_contains(m3_unicode_number_shards,
                             m3_unicode_number_shard_count, code_point);
}

bool m3_unicode_is_whitespace(uint32_t code_point)
{
    return m3_range_contains(m3_unicode_whitespace_shards,
                             m3_unicode_whitespace_shard_count, code_point);
}

static uint8_t m3_unicode_ccc_value(uint32_t code_point)
{
    size_t shard_index;

    for (shard_index = 0U; shard_index < m3_unicode_ccc_shard_count;
         ++shard_index) {
        const m3_unicode_ccc *items = m3_unicode_ccc_shards[shard_index].items;
        size_t count = m3_unicode_ccc_shards[shard_index].count;
        size_t low = 0U;
        size_t high = count;

        if (count == 0U || code_point < items[0].code_point ||
            code_point > items[count - 1U].code_point) {
            continue;
        }
        while (low < high) {
            size_t middle = low + (high - low) / 2U;

            if (code_point < items[middle].code_point) {
                high = middle;
            } else if (code_point > items[middle].code_point) {
                low = middle + 1U;
            } else {
                return items[middle].canonical_combining_class;
            }
        }
    }
    return 0U;
}

uint32_t m3_unicode_simple_fold(uint32_t code_point)
{
    size_t shard_index;

    if (code_point >= (uint32_t)'A' && code_point <= (uint32_t)'Z') {
        return code_point + ((uint32_t)'a' - (uint32_t)'A');
    }
    for (shard_index = 0U; shard_index < m3_unicode_fold_shard_count;
         ++shard_index) {
        const m3_unicode_fold *items = m3_unicode_fold_shards[shard_index].items;
        size_t count = m3_unicode_fold_shards[shard_index].count;
        size_t low = 0U;
        size_t high = count;

        while (low < high) {
            size_t middle = low + (high - low) / 2U;

            if (code_point < items[middle].code_point) {
                high = middle;
            } else if (code_point > items[middle].code_point) {
                low = middle + 1U;
            } else {
                return items[middle].folded;
            }
        }
    }
    return code_point;
}

bool m3_utf8_decode(const uint8_t *data, size_t size, uint32_t *code_point,
                    size_t *length)
{
    uint32_t value;
    size_t expected;
    size_t index;

    if (data == NULL || code_point == NULL || length == NULL || size == 0U) {
        return false;
    }
    if (data[0] < 0x80U) {
        *code_point = data[0];
        *length = 1U;
        return true;
    }
    if (data[0] >= 0xC2U && data[0] <= 0xDFU) {
        value = (uint32_t)(data[0] & 0x1FU);
        expected = 2U;
    } else if (data[0] >= 0xE0U && data[0] <= 0xEFU) {
        value = (uint32_t)(data[0] & 0x0FU);
        expected = 3U;
    } else if (data[0] >= 0xF0U && data[0] <= 0xF4U) {
        value = (uint32_t)(data[0] & 0x07U);
        expected = 4U;
    } else {
        return false;
    }
    if (size < expected) {
        return false;
    }
    for (index = 1U; index < expected; ++index) {
        if ((data[index] & 0xC0U) != 0x80U) {
            return false;
        }
        value = (value << 6U) | (uint32_t)(data[index] & 0x3FU);
    }
    if ((expected == 3U && value < 0x800U) ||
        (expected == 4U && value < 0x10000U) || value > 0x10FFFFU ||
        (value >= 0xD800U && value <= 0xDFFFU)) {
        return false;
    }
    *code_point = value;
    *length = expected;
    return true;
}

size_t m3_utf8_encode(uint32_t code_point, uint8_t output[4])
{
    if (output == NULL || code_point > 0x10FFFFU ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
        return 0U;
    }
    if (code_point <= 0x7FU) {
        output[0] = (uint8_t)code_point;
        return 1U;
    }
    if (code_point <= 0x7FFU) {
        output[0] = (uint8_t)(0xC0U | (code_point >> 6U));
        output[1] = (uint8_t)(0x80U | (code_point & 0x3FU));
        return 2U;
    }
    if (code_point <= 0xFFFFU) {
        output[0] = (uint8_t)(0xE0U | (code_point >> 12U));
        output[1] = (uint8_t)(0x80U | ((code_point >> 6U) & 0x3FU));
        output[2] = (uint8_t)(0x80U | (code_point & 0x3FU));
        return 3U;
    }
    output[0] = (uint8_t)(0xF0U | (code_point >> 18U));
    output[1] = (uint8_t)(0x80U | ((code_point >> 12U) & 0x3FU));
    output[2] = (uint8_t)(0x80U | ((code_point >> 6U) & 0x3FU));
    output[3] = (uint8_t)(0x80U | (code_point & 0x3FU));
    return 4U;
}

static const m3_unicode_decomposition *m3_find_decomposition(uint32_t cp)
{
    size_t shard_index;

    for (shard_index = 0U; shard_index < m3_unicode_decomposition_shard_count;
         ++shard_index) {
        const m3_unicode_decomposition *items =
            m3_unicode_decomposition_shards[shard_index].items;
        size_t count = m3_unicode_decomposition_shards[shard_index].count;
        size_t low = 0U;
        size_t high = count;

        if (count == 0U || cp < items[0].code_point ||
            cp > items[count - 1U].code_point) {
            continue;
        }
        while (low < high) {
            size_t middle = low + (high - low) / 2U;

            if (cp < items[middle].code_point) {
                high = middle;
            } else if (cp > items[middle].code_point) {
                low = middle + 1U;
            } else {
                return &items[middle];
            }
        }
    }
    return NULL;
}

static m3_status m3_code_points_append(m3_code_points *points, uint32_t cp,
                                        m3_error *error)
{
    if (points->count == points->capacity) {
        size_t capacity = points->capacity == 0U ? 64U : points->capacity * 2U;
        uint32_t *items;

        if (capacity < points->capacity ||
            capacity > SIZE_MAX / sizeof(*points->items)) {
            return m3_error_set(error, M3_STATUS_OVERFLOW,
                                "normalized text is too large");
        }
        items = realloc(points->items, capacity * sizeof(*points->items));
        if (items == NULL) {
            return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                "cannot allocate normalized text");
        }
        points->items = items;
        points->capacity = capacity;
    }
    points->items[points->count++] = cp;
    return M3_STATUS_OK;
}

static m3_status m3_decompose(m3_code_points *points, uint32_t cp,
                               unsigned int depth, m3_error *error)
{
    const m3_unicode_decomposition *decomposition;

    if (depth >= 64U) {
        return m3_error_set(error, M3_STATUS_INTERNAL,
                            "Unicode decomposition depth exceeded");
    }
    if (cp >= M3_HANGUL_S_BASE &&
        cp < M3_HANGUL_S_BASE + M3_HANGUL_S_COUNT) {
        uint32_t index = cp - M3_HANGUL_S_BASE;
        uint32_t trailing = index % M3_HANGUL_T_COUNT;
        m3_status status;

        status = m3_code_points_append(
            points, M3_HANGUL_L_BASE + index / M3_HANGUL_N_COUNT, error);
        if (status == M3_STATUS_OK) {
            status = m3_code_points_append(
                points, M3_HANGUL_V_BASE +
                            (index % M3_HANGUL_N_COUNT) / M3_HANGUL_T_COUNT,
                error);
        }
        if (status == M3_STATUS_OK && trailing != 0U) {
            status = m3_code_points_append(points,
                                            M3_HANGUL_T_BASE + trailing,
                                            error);
        }
        return status;
    }
    decomposition = m3_find_decomposition(cp);
    if (decomposition == NULL) {
        return m3_code_points_append(points, cp, error);
    }
    {
        m3_status status = m3_decompose(points, decomposition->first,
                                        depth + 1U, error);

        if (status == M3_STATUS_OK && decomposition->second != 0U) {
            status = m3_decompose(points, decomposition->second,
                                  depth + 1U, error);
        }
        return status;
    }
}

static uint32_t m3_compose_pair(uint32_t first, uint32_t second)
{
    size_t shard_index;

    if (first >= M3_HANGUL_L_BASE &&
        first < M3_HANGUL_L_BASE + M3_HANGUL_L_COUNT &&
        second >= M3_HANGUL_V_BASE &&
        second < M3_HANGUL_V_BASE + M3_HANGUL_V_COUNT) {
        return M3_HANGUL_S_BASE +
               ((first - M3_HANGUL_L_BASE) * M3_HANGUL_V_COUNT +
                (second - M3_HANGUL_V_BASE)) * M3_HANGUL_T_COUNT;
    }
    if (first >= M3_HANGUL_S_BASE &&
        first < M3_HANGUL_S_BASE + M3_HANGUL_S_COUNT &&
        (first - M3_HANGUL_S_BASE) % M3_HANGUL_T_COUNT == 0U &&
        second > M3_HANGUL_T_BASE &&
        second < M3_HANGUL_T_BASE + M3_HANGUL_T_COUNT) {
        return first + second - M3_HANGUL_T_BASE;
    }
    for (shard_index = 0U; shard_index < m3_unicode_composition_shard_count;
         ++shard_index) {
        const m3_unicode_composition *items =
            m3_unicode_composition_shards[shard_index].items;
        size_t count = m3_unicode_composition_shards[shard_index].count;
        size_t low = 0U;
        size_t high = count;

        while (low < high) {
            size_t middle = low + (high - low) / 2U;
            const m3_unicode_composition *item = &items[middle];

            if (first < item->first ||
                (first == item->first && second < item->second)) {
                high = middle;
            } else if (first > item->first || second > item->second) {
                low = middle + 1U;
            } else {
                return item->composed;
            }
        }
    }
    return 0U;
}

static void m3_reorder(m3_code_points *points)
{
    size_t index;

    for (index = 1U; index < points->count; ++index) {
        uint8_t current_class = m3_unicode_ccc_value(points->items[index]);
        size_t position = index;

        if (current_class == 0U) {
            continue;
        }
        while (position > 0U) {
            uint8_t previous_class =
                m3_unicode_ccc_value(points->items[position - 1U]);

            if (previous_class == 0U || previous_class <= current_class) {
                break;
            }
            {
                uint32_t temporary = points->items[position - 1U];
                points->items[position - 1U] = points->items[position];
                points->items[position] = temporary;
            }
            position -= 1U;
        }
    }
}

static void m3_compose(m3_code_points *points)
{
    size_t starter_position = 0U;
    uint32_t starter;
    uint8_t last_class = 0U;
    size_t read;
    size_t write;

    if (points->count == 0U) {
        return;
    }
    starter = points->items[0];
    write = 1U;
    for (read = 1U; read < points->count; ++read) {
        uint32_t current = points->items[read];
        uint8_t current_class = m3_unicode_ccc_value(current);
        uint32_t composed = m3_compose_pair(starter, current);

        if (composed != 0U && (last_class == 0U || last_class < current_class)) {
            points->items[starter_position] = composed;
            starter = composed;
            continue;
        }
        if (current_class == 0U) {
            starter_position = write;
            starter = current;
        }
        last_class = current_class;
        points->items[write++] = current;
    }
    points->count = write;
}

m3_status m3_unicode_nfc(const uint8_t *data, size_t size, uint8_t **output,
                         size_t *output_size, m3_error *error)
{
    m3_code_points points = {0};
    uint8_t *bytes = NULL;
    size_t position = 0U;
    size_t byte_count = 0U;
    m3_status status = M3_STATUS_OK;
    size_t index;

    if (output == NULL || output_size == NULL || (size != 0U && data == NULL)) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "NFC argument is invalid");
    }
    *output = NULL;
    *output_size = 0U;
    while (position < size) {
        uint32_t cp;
        size_t length;

        if (!m3_utf8_decode(data + position, size - position, &cp, &length)) {
            status = m3_error_set(error, M3_STATUS_INVALID_FORMAT,
                                  "invalid UTF-8 at byte %zu", position);
            break;
        }
        status = m3_decompose(&points, cp, 0U, error);
        if (status != M3_STATUS_OK) {
            break;
        }
        position += length;
    }
    if (status == M3_STATUS_OK) {
        m3_reorder(&points);
        m3_compose(&points);
        if (points.count > (SIZE_MAX - 1U) / 4U) {
            status = m3_error_set(error, M3_STATUS_OVERFLOW,
                                  "normalized text byte count overflows");
        } else {
            bytes = malloc(points.count * 4U + 1U);
            if (bytes == NULL) {
                free(points.items);
                return m3_error_set(error, M3_STATUS_OUT_OF_MEMORY,
                                    "cannot allocate normalized UTF-8");
            }
        }
    }
    if (status == M3_STATUS_OK) {
        for (index = 0U; index < points.count; ++index) {
            byte_count += m3_utf8_encode(points.items[index],
                                         bytes + byte_count);
        }
        bytes[byte_count] = 0U;
        *output = bytes;
        *output_size = byte_count;
        m3_error_reset(error);
    } else {
        free(bytes);
    }
    free(points.items);
    return status;
}
