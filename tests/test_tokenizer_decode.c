/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "tokenizer_test.h"

#include <stdlib.h>
#include <string.h>

static void m3_expect_decoding(m3_test_context *test,
                               m3_tokenizer *tokenizer,
                               const uint32_t *ids, size_t count,
                               bool skip_special,
                               const uint8_t *expected,
                               size_t expected_length,
                               const char *message)
{
    m3_tokenizer_text text;
    m3_error error;
    m3_status status;

    m3_tokenizer_text_init(&text);
    status = m3_tokenizer_decode(tokenizer, ids, count, skip_special,
                                 &text, &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_OK, message);
    if (status == M3_STATUS_OK) {
        M3_TEST_EXPECT(test,
                       text.length == expected_length &&
                           (expected_length == 0U ||
                            memcmp(text.data, expected, expected_length) == 0),
                       message);
        M3_TEST_EXPECT(test, text.data != NULL && text.data[text.length] == 0U,
                       "decoded convenience terminator");
    }
    m3_tokenizer_text_dispose(&text);
}

void m3_test_tokenizer_added_and_decode(m3_test_context *test)
{
    static const uint32_t added_ids[] = {
        M3_TOKEN_AUDIO_CFG, 151657U,
    };
    static const uint32_t incomplete[] = {0xF0U, 0x9FU};
    static const uint32_t invalid_third[] = {0xE2U, 0x82U, (uint32_t)'A'};
    static const uint32_t invalid_second[] = {0xE2U, (uint32_t)'(', 0xA1U};
    static const uint8_t replacement[] = {0xEFU, 0xBFU, 0xBDU};
    static const uint8_t replacement_a[] = {0xEFU, 0xBFU, 0xBDU, 'A'};
    static const uint8_t replacement_paren_replacement[] = {
        0xEFU, 0xBFU, 0xBDU, '(', 0xEFU, 0xBFU, 0xBDU,
    };
    static const char all_added[] =
        "<|audio_cfg|><ordinary_151657>";
    static const char ordinary_added[] = "<ordinary_151657>";
    m3_tokenizer tokenizer;
    m3_tokenizer_fixture fixture;
    m3_token_ids encoded;
    m3_tokenizer_text preserved;
    m3_error error;

    m3_tokenizer_init(&tokenizer);
    m3_token_ids_init(&encoded);
    m3_tokenizer_text_init(&preserved);
    if (m3_test_tokenizer_fixture_load(
            &tokenizer, &fixture, M3_TOKENIZER_FIXTURE_VALID, &error)) {
        static const uint32_t audio_cfg[] = {M3_TOKEN_AUDIO_CFG};
        uint32_t invalid_id = M3_TOKENIZER_SEMANTIC_FIRST;
        uint8_t *saved;
        size_t saved_length;

        M3_TEST_EXPECT(
            test,
            m3_test_tokenizer_encode_text(
                &tokenizer, "<|audio_cfg|>", sizeof("<|audio_cfg|>") - 1U,
                &encoded, &error),
            "encode atomic audio CFG marker");
        m3_test_expect_ids(test, &encoded, audio_cfg, 1U,
                           "atomic audio CFG marker ID");
        m3_expect_decoding(test, &tokenizer, NULL, 0U, false, NULL, 0U,
                           "empty decode");
        m3_expect_decoding(test, &tokenizer, added_ids,
                           sizeof(added_ids) / sizeof(added_ids[0]), false,
                           (const uint8_t *)all_added,
                           sizeof(all_added) - 1U,
                           "decode special and ordinary added tokens");
        m3_expect_decoding(test, &tokenizer, added_ids,
                           sizeof(added_ids) / sizeof(added_ids[0]), true,
                           (const uint8_t *)ordinary_added,
                           sizeof(ordinary_added) - 1U,
                           "skip only special added tokens");
        m3_expect_decoding(test, &tokenizer, incomplete,
                           sizeof(incomplete) / sizeof(incomplete[0]), false,
                           replacement, sizeof(replacement),
                           "incomplete UTF-8 tail is one replacement");
        m3_expect_decoding(test, &tokenizer, invalid_third,
                           sizeof(invalid_third) / sizeof(invalid_third[0]),
                           false, replacement_a, sizeof(replacement_a),
                           "invalid third byte replacement grouping");
        m3_expect_decoding(
            test, &tokenizer, invalid_second,
            sizeof(invalid_second) / sizeof(invalid_second[0]), false,
            replacement_paren_replacement,
            sizeof(replacement_paren_replacement),
            "invalid continuation replacement grouping");

        M3_TEST_EXPECT(test,
                       m3_tokenizer_decode(&tokenizer, added_ids, 2U, false,
                                           &preserved, &error) == M3_STATUS_OK,
                       "seed atomic decode output");
        saved = preserved.data;
        saved_length = preserved.length;
        M3_TEST_EXPECT(test,
                       m3_tokenizer_decode(&tokenizer, &invalid_id, 1U, false,
                                           &preserved, &error) ==
                           M3_STATUS_OUT_OF_RANGE,
                       "reject semantic ID during decode");
        M3_TEST_EXPECT(test,
                       preserved.data == saved &&
                           preserved.length == saved_length &&
                           memcmp(preserved.data, all_added,
                                  sizeof(all_added) - 1U) == 0,
                       "decode failure preserves output");
        M3_TEST_EXPECT(test,
                       m3_tokenizer_decode(&tokenizer, added_ids, SIZE_MAX,
                                           false, &preserved, &error) ==
                           M3_STATUS_OVERFLOW,
                       "reject overflowing decode count");
    } else {
        M3_TEST_EXPECT(test, false, "load decode fixture");
    }
    m3_tokenizer_text_dispose(&preserved);
    m3_token_ids_dispose(&encoded);
    m3_tokenizer_fixture_dispose(&fixture);
    m3_tokenizer_dispose(&tokenizer);
}

void m3_test_tokenizer_cfg_row(m3_test_context *test)
{
    static const uint32_t prompt[] = {
        M3_TOKEN_IM_START, 4U, 5U, 6U, M3_TOKEN_IM_END,
        M3_TOKEN_AUDIO_START,
    };
    static const uint32_t expected[] = {
        M3_TOKEN_IM_START, M3_TOKEN_AUDIO_CFG, M3_TOKEN_AUDIO_CFG,
        M3_TOKEN_AUDIO_CFG, M3_TOKEN_IM_END, M3_TOKEN_AUDIO_START,
    };
    static const uint32_t minimal[] = {
        M3_TOKEN_IM_START, M3_TOKEN_IM_END, M3_TOKEN_AUDIO_START,
    };
    m3_token_ids cfg;
    m3_error error;
    uint32_t invalid[sizeof(prompt) / sizeof(prompt[0])];
    uint32_t *saved;
    size_t index;

    m3_token_ids_init(&cfg);
    M3_TEST_EXPECT(test,
                   m3_tokenizer_build_cfg_row(
                       prompt, sizeof(prompt) / sizeof(prompt[0]), &cfg,
                       &error) == M3_STATUS_OK,
                   "build CFG row");
    m3_test_expect_ids(test, &cfg, expected,
                       sizeof(expected) / sizeof(expected[0]),
                       "CFG middle fill");
    saved = cfg.data;

    for (index = 0U; index < 3U; ++index) {
        size_t boundary = index == 0U ? 0U
                          : index == 1U ? sizeof(prompt) / sizeof(prompt[0]) - 2U
                                       : sizeof(prompt) / sizeof(prompt[0]) - 1U;

        (void)memcpy(invalid, prompt, sizeof(prompt));
        invalid[boundary] = 0U;
        M3_TEST_EXPECT(test,
                       m3_tokenizer_build_cfg_row(
                           invalid, sizeof(invalid) / sizeof(invalid[0]),
                           &cfg, &error) == M3_STATUS_INVALID_FORMAT,
                       "reject wrong CFG boundary token");
        M3_TEST_EXPECT(test,
                       cfg.data == saved && cfg.count ==
                           sizeof(expected) / sizeof(expected[0]) &&
                           memcmp(cfg.data, expected, sizeof(expected)) == 0,
                       "CFG boundary failure preserves output");
    }
    M3_TEST_EXPECT(test,
                   m3_tokenizer_build_cfg_row(minimal, 3U, &cfg, &error) ==
                       M3_STATUS_OK,
                   "minimal CFG row");
    m3_test_expect_ids(test, &cfg, minimal, 3U,
                       "minimal CFG preserves all boundaries");
    saved = cfg.data;
    M3_TEST_EXPECT(test,
                   m3_tokenizer_build_cfg_row(minimal, 2U, &cfg, &error) ==
                       M3_STATUS_OUT_OF_RANGE,
                   "reject short CFG row");
    M3_TEST_EXPECT(test, cfg.data == saved && cfg.count == 3U,
                   "short CFG failure preserves output");
    m3_token_ids_dispose(&cfg);
}
