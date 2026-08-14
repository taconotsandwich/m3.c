/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_tokenizer_internal.h"
#include "tokenizer_test.h"

#include <stdlib.h>
#include <string.h>

static void m3_expect_bad_kind(m3_test_context *test,
                               m3_tokenizer *tokenizer,
                               m3_tokenizer_fixture_kind kind,
                               bool null_error,
                               const char *message)
{
    m3_tokenizer_fixture fixture;
    m3_error error;
    bool loaded = m3_test_tokenizer_fixture_load(
        tokenizer, &fixture, kind, null_error ? NULL : &error);

    M3_TEST_EXPECT(test, !loaded, message);
    m3_tokenizer_fixture_dispose(&fixture);
}

static void m3_expect_trailing_comma(m3_test_context *test,
                                     m3_tokenizer *tokenizer,
                                     const char *needle,
                                     const char *message)
{
    m3_tokenizer probe;
    m3_tokenizer_fixture fixture;
    m3_error error;
    bool built;
    m3_status status = M3_STATUS_INTERNAL;

    m3_tokenizer_init(&probe);
    built = m3_test_tokenizer_fixture_load(
        &probe, &fixture, M3_TOKENIZER_FIXTURE_VALID, &error);
    M3_TEST_EXPECT(test, built, "build trailing-comma fixture");
    if (built) {
        M3_TEST_EXPECT(test,
                       m3_test_fixture_insert_before(&fixture, needle,
                                                     (uint8_t)','),
                       "mutate trailing-comma fixture");
        status = m3_tokenizer_load_data(tokenizer, fixture.data,
                                        fixture.size, &fixture.contract,
                                        &error);
        M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_FORMAT, message);
    }
    m3_tokenizer_fixture_dispose(&fixture);
    m3_tokenizer_dispose(&probe);
}

void m3_test_tokenizer_loader_rejection(m3_test_context *test)
{
    static const m3_tokenizer_fixture_kind malformed_kinds[] = {
        M3_TOKENIZER_FIXTURE_DUPLICATE_VOCAB,
        M3_TOKENIZER_FIXTURE_DUPLICATE_MERGE,
        M3_TOKENIZER_FIXTURE_MISSING_MARKER,
        M3_TOKENIZER_FIXTURE_WRONG_ADDED_ID,
    };
    static const m3_tokenizer_fixture_kind null_error_kinds[] = {
        M3_TOKENIZER_FIXTURE_EMPTY_PATTERN,
        M3_TOKENIZER_FIXTURE_EMPTY_MODEL_FIELD,
        M3_TOKENIZER_FIXTURE_EMPTY_VOCAB_TOKEN,
        M3_TOKENIZER_FIXTURE_EMPTY_MERGE_TOKEN,
    };
    static const uint8_t malformed_json[] = {'{', '"', 'x', '"', ':', '}'};
    static const uint8_t root_trailing[] =
        "{\"version\":\"1.0\",}";
    static const uint32_t expected[] = {M3_FIXTURE_AA, M3_FIXTURE_AA};
    m3_tokenizer tokenizer;
    m3_tokenizer_fixture valid;
    m3_token_ids ids;
    m3_error error;
    size_t index;

    m3_tokenizer_init(&tokenizer);
    m3_token_ids_init(&ids);
    M3_TEST_EXPECT(test,
                   m3_test_tokenizer_fixture_load(
                       &tokenizer, &valid, M3_TOKENIZER_FIXTURE_VALID,
                       &error),
                   "load valid tokenizer before atomic failures");

    for (index = 0U;
         index < sizeof(malformed_kinds) / sizeof(malformed_kinds[0]);
         ++index) {
        m3_expect_bad_kind(test, &tokenizer, malformed_kinds[index], false,
                           "reject malformed tokenizer fixture");
    }
    for (index = 0U;
         index < sizeof(null_error_kinds) / sizeof(null_error_kinds[0]);
         ++index) {
        m3_expect_bad_kind(test, &tokenizer, null_error_kinds[index], true,
                           "reject empty tokenizer field without error sink");
    }
    M3_TEST_EXPECT(test,
                   m3_tokenizer_load_data(
                       &tokenizer, malformed_json, sizeof(malformed_json),
                       &valid.contract, &error) == M3_STATUS_INVALID_FORMAT,
                   "reject malformed JSON");
    M3_TEST_EXPECT(test,
                   m3_tokenizer_load_data(
                       &tokenizer, root_trailing, sizeof(root_trailing) - 1U,
                       &valid.contract, NULL) == M3_STATUS_INVALID_FORMAT,
                   "reject root trailing comma without error sink");

    m3_expect_trailing_comma(test, &tokenizer, "],\"model\"",
                             "reject added-token array trailing comma");
    m3_expect_trailing_comma(test, &tokenizer, "},\"merges\"",
                             "reject vocabulary object trailing comma");
    m3_expect_trailing_comma(test, &tokenizer, "]}}",
                             "reject merge array trailing comma");
    m3_expect_trailing_comma(test, &tokenizer,
                             "},\"pre_tokenizer\"",
                             "reject nested object trailing comma");

    M3_TEST_EXPECT(test,
                   m3_test_tokenizer_encode_text(&tokenizer, "aaaa", 4U,
                                                 &ids, &error),
                   "old tokenizer remains usable after failed reloads");
    m3_test_expect_ids(test, &ids, expected,
                       sizeof(expected) / sizeof(expected[0]),
                       "atomic reload preserves old tokenizer state");
    {
        static const uint8_t invalid_utf8[] = {0xC0U};
        uint32_t *saved = ids.data;
        size_t saved_count = ids.count;
        char *many_digits = malloc(M3_TOKENIZER_MAX_PROMPT_IDS + 1U);

        M3_TEST_EXPECT(test,
                       m3_tokenizer_encode(
                           &tokenizer,
                           (m3_tokenizer_bytes){invalid_utf8,
                                                sizeof(invalid_utf8)},
                           &ids, &error) == M3_STATUS_INVALID_FORMAT,
                       "reject invalid input UTF-8");
        M3_TEST_EXPECT(test, ids.data == saved && ids.count == saved_count,
                       "invalid UTF-8 preserves token output");
        M3_TEST_EXPECT(test, many_digits != NULL,
                       "allocate output-overflow input");
        if (many_digits != NULL) {
            (void)memset(many_digits, '1',
                         M3_TOKENIZER_MAX_PROMPT_IDS + 1U);
            M3_TEST_EXPECT(
                test,
                m3_tokenizer_encode(
                    &tokenizer,
                    (m3_tokenizer_bytes){
                        (const uint8_t *)many_digits,
                        M3_TOKENIZER_MAX_PROMPT_IDS + 1U},
                    &ids, &error) == M3_STATUS_OUT_OF_RANGE,
                "reject token output beyond prompt limit");
            M3_TEST_EXPECT(test,
                           ids.data == saved && ids.count == saved_count,
                           "token overflow preserves output");
        }
        free(many_digits);
    }

    m3_token_ids_dispose(&ids);
    m3_tokenizer_fixture_dispose(&valid);
    m3_tokenizer_dispose(&tokenizer);
}
