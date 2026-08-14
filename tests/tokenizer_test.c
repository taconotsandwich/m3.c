/* SPDX-License-Identifier: GPL-2.0-only */

#include "tokenizer_test.h"

#include "m3_tokenizer_internal.h"

#include <stdlib.h>
#include <string.h>

static const m3_fixture_merge m3_test_merges[] = {
    {(uint32_t)'a', (uint32_t)'a'},
    {M3_FIXTURE_AA, (uint32_t)'a'},
    {(uint32_t)'a', (uint32_t)'1'},
    {(uint32_t)' ', (uint32_t)' '},
    {(uint32_t)' ', (uint32_t)'h'},
    {(uint32_t)' ', (uint32_t)'!'},
    {(uint32_t)'\t', (uint32_t)'h'},
    {(uint32_t)'\t', (uint32_t)'!'},
    {(uint32_t)'1', (uint32_t)'2'},
    {(uint32_t)'\r', (uint32_t)'\n'},
    {0xC3U, 0xA9U},
    {(uint32_t)'I', (uint32_t)'\''},
    {(uint32_t)'\'', 0xC5U},
    {M3_FIXTURE_APOSTROPHE_C5, 0xBFU},
    {M3_FIXTURE_APOSTROPHE_LONG_S, (uint32_t)'a'},
    {(uint32_t)'\'', (uint32_t)'M'},
    {M3_FIXTURE_APOSTROPHE_M, (uint32_t)'x'},
    {0xD9U, 0xA1U},
    {0xD9U, 0xA2U},
    {M3_FIXTURE_ARABIC_ONE, M3_FIXTURE_ARABIC_TWO},
    {0xC2U, 0xA0U},
    {M3_FIXTURE_NBSP, (uint32_t)'h'},
    {M3_FIXTURE_NBSP, (uint32_t)'!'},
};

bool m3_test_tokenizer_fixture_load(m3_tokenizer *tokenizer,
                                    m3_tokenizer_fixture *fixture,
                                    m3_tokenizer_fixture_kind kind,
                                    m3_error *error)
{
    if (!m3_tokenizer_fixture_build(
            fixture, m3_test_merges,
            sizeof(m3_test_merges) / sizeof(m3_test_merges[0]), kind)) {
        return false;
    }
    return m3_tokenizer_load_data(tokenizer, fixture->data, fixture->size,
                                  &fixture->contract, error) == M3_STATUS_OK;
}

bool m3_test_tokenizer_encode_text(m3_tokenizer *tokenizer,
                                   const char *text, size_t length,
                                   m3_token_ids *ids, m3_error *error)
{
    return m3_tokenizer_encode(
               tokenizer,
               (m3_tokenizer_bytes){(const uint8_t *)text, length},
               ids, error) == M3_STATUS_OK;
}

void m3_test_expect_ids(m3_test_context *test, const m3_token_ids *actual,
                        const uint32_t *expected, size_t expected_count,
                        const char *message)
{
    bool equal = actual != NULL && actual->count == expected_count;

    if (equal && expected_count != 0U) {
        equal = expected != NULL &&
                memcmp(actual->data, expected,
                       expected_count * sizeof(*expected)) == 0;
    }
    M3_TEST_EXPECT(test, equal, message);
}

bool m3_test_fixture_insert_before(m3_tokenizer_fixture *fixture,
                                   const char *needle, uint8_t byte)
{
    size_t needle_length;
    size_t position;
    uint8_t *data;

    if (fixture == NULL || fixture->data == NULL || needle == NULL ||
        fixture->size == SIZE_MAX) {
        return false;
    }
    needle_length = strlen(needle);
    if (needle_length == 0U || needle_length > fixture->size) {
        return false;
    }
    for (position = 0U; position + needle_length <= fixture->size;
         ++position) {
        if (memcmp(fixture->data + position, needle, needle_length) == 0) {
            break;
        }
    }
    if (position + needle_length > fixture->size) {
        return false;
    }
    data = realloc(fixture->data, fixture->size + 1U);
    if (data == NULL) {
        return false;
    }
    fixture->data = data;
    (void)memmove(data + position + 1U, data + position,
                  fixture->size - position);
    data[position] = byte;
    fixture->size += 1U;
    return true;
}
