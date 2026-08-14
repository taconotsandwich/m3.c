/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "tokenizer_test.h"

#include <string.h>

static void m3_expect_encoding(m3_test_context *test,
                               m3_tokenizer *tokenizer,
                               const char *text, size_t length,
                               const uint32_t *expected,
                               size_t expected_count,
                               const char *message)
{
    m3_token_ids ids;
    m3_error error;
    bool encoded;

    m3_token_ids_init(&ids);
    encoded = m3_test_tokenizer_encode_text(tokenizer, text, length,
                                            &ids, &error);
    M3_TEST_EXPECT(test, encoded, message);
    if (encoded) {
        m3_test_expect_ids(test, &ids, expected, expected_count, message);
    }
    m3_token_ids_dispose(&ids);
}

static bool m3_load_encoding_fixture(m3_test_context *test,
                                     m3_tokenizer *tokenizer,
                                     m3_tokenizer_fixture *fixture)
{
    m3_error error;
    bool loaded;

    m3_tokenizer_init(tokenizer);
    loaded = m3_test_tokenizer_fixture_load(
        tokenizer, fixture, M3_TOKENIZER_FIXTURE_VALID, &error);
    M3_TEST_EXPECT(test, loaded, "load synthetic tokenizer fixture");
    return loaded;
}

void m3_test_tokenizer_bpe_boundaries(m3_test_context *test)
{
    static const uint32_t aaaa[] = {M3_FIXTURE_AA, M3_FIXTURE_AA};
    static const uint32_t a1[] = {(uint32_t)'a', (uint32_t)'1'};
    static const uint32_t digits[] = {(uint32_t)'1', (uint32_t)'2'};
    static const uint32_t contraction[] = {
        (uint32_t)'I', M3_FIXTURE_APOSTROPHE_M,
    };
    static const uint32_t long_s[] = {
        M3_FIXTURE_APOSTROPHE_LONG_S, (uint32_t)'a',
    };
    static const uint32_t upper_m[] = {
        M3_FIXTURE_APOSTROPHE_M, (uint32_t)'x',
    };
    m3_tokenizer tokenizer;
    m3_tokenizer_fixture fixture;

    if (m3_load_encoding_fixture(test, &tokenizer, &fixture)) {
        m3_expect_encoding(test, &tokenizer, "", 0U, NULL, 0U,
                           "empty input");
        m3_expect_encoding(test, &tokenizer, "aaaa", 4U, aaaa,
                           sizeof(aaaa) / sizeof(aaaa[0]),
                           "leftmost overlapping merge");
        m3_expect_encoding(test, &tokenizer, "a1", 2U, a1,
                           sizeof(a1) / sizeof(a1[0]),
                           "letter-number boundary");
        m3_expect_encoding(test, &tokenizer, "12", 2U, digits,
                           sizeof(digits) / sizeof(digits[0]),
                           "digit isolation");
        m3_expect_encoding(test, &tokenizer, "I'M", 3U, contraction,
                           sizeof(contraction) / sizeof(contraction[0]),
                           "contraction does not merge with prior word");
        m3_expect_encoding(test, &tokenizer, "'\xC5\xBF" "a", 4U,
                           long_s, sizeof(long_s) / sizeof(long_s[0]),
                           "long-s case-insensitive contraction");
        m3_expect_encoding(test, &tokenizer, "'Mx", 3U, upper_m,
                           sizeof(upper_m) / sizeof(upper_m[0]),
                           "uppercase contraction boundary");
    }
    m3_tokenizer_fixture_dispose(&fixture);
    m3_tokenizer_dispose(&tokenizer);
}

void m3_test_tokenizer_unicode_vectors(m3_test_context *test)
{
    static const uint32_t acute[] = {M3_FIXTURE_E_ACUTE};
    static const uint32_t chinese[] = {
        0xE4U, 0xBDU, 0xA0U, 0xE5U, 0xA5U, 0xBDU,
    };
    static const uint32_t emoji[] = {0xF0U, 0x9FU, 0x99U, 0x82U};
    static const uint32_t crlf[] = {
        (uint32_t)'a', M3_FIXTURE_CRLF, (uint32_t)'b', (uint32_t)'\n',
    };
    static const uint32_t hangul[] = {0xEAU, 0xB0U, 0x81U};
    static const uint32_t angstrom[] = {0xC3U, 0x85U};
    static const uint32_t reordered[] = {0xC3U, 0xA0U, 0xCCU, 0x95U};
    static const uint32_t blocked[] = {
        (uint32_t)'A', 0xCCU, 0x85U, 0xCCU, 0x80U,
    };
    static const uint32_t arabic_digits[] = {
        M3_FIXTURE_ARABIC_ONE, M3_FIXTURE_ARABIC_TWO,
    };
    static const uint32_t nbsp_letter[] = {
        M3_FIXTURE_NBSP, M3_FIXTURE_NBSP_H, (uint32_t)'i',
    };
    static const uint32_t nbsp_symbol[] = {
        M3_FIXTURE_NBSP, (uint32_t)'!',
    };
    static const char decomposed[] = "e\xCC\x81";
    static const char precomposed[] = "\xC3\xA9";
    static const char nihao[] = "\xE4\xBD\xA0\xE5\xA5\xBD";
    static const char smile[] = "\xF0\x9F\x99\x82";
    static const char hangul_jamo[] =
        "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8";
    static const char angstrom_sign[] = "\xE2\x84\xAB";
    static const char reorder_input[] = "a\xCC\x95\xCC\x80";
    static const char blocked_input[] = "A\xCC\x85\xCC\x80";
    static const char arabic_input[] = "\xD9\xA1\xD9\xA2";
    static const char nbsp_hi[] = "\xC2\xA0\xC2\xA0hi";
    static const char nbsp_bang[] = "\xC2\xA0!";
    m3_tokenizer tokenizer;
    m3_tokenizer_fixture fixture;

    if (m3_load_encoding_fixture(test, &tokenizer, &fixture)) {
        m3_expect_encoding(test, &tokenizer, decomposed,
                           sizeof(decomposed) - 1U, acute,
                           sizeof(acute) / sizeof(acute[0]),
                           "NFC decomposed acute");
        m3_expect_encoding(test, &tokenizer, precomposed,
                           sizeof(precomposed) - 1U, acute,
                           sizeof(acute) / sizeof(acute[0]),
                           "NFC precomposed acute");
        m3_expect_encoding(test, &tokenizer, nihao, sizeof(nihao) - 1U,
                           chinese, sizeof(chinese) / sizeof(chinese[0]),
                           "Chinese byte-level vector");
        m3_expect_encoding(test, &tokenizer, smile, sizeof(smile) - 1U,
                           emoji, sizeof(emoji) / sizeof(emoji[0]),
                           "emoji byte-level vector");
        m3_expect_encoding(test, &tokenizer, "a\r\nb\n", 5U, crlf,
                           sizeof(crlf) / sizeof(crlf[0]),
                           "CR LF scanner vector");
        m3_expect_encoding(test, &tokenizer, hangul_jamo,
                           sizeof(hangul_jamo) - 1U, hangul,
                           sizeof(hangul) / sizeof(hangul[0]),
                           "NFC Hangul algorithmic composition");
        m3_expect_encoding(test, &tokenizer, angstrom_sign,
                           sizeof(angstrom_sign) - 1U, angstrom,
                           sizeof(angstrom) / sizeof(angstrom[0]),
                           "NFC canonical singleton");
        m3_expect_encoding(test, &tokenizer, reorder_input,
                           sizeof(reorder_input) - 1U, reordered,
                           sizeof(reordered) / sizeof(reordered[0]),
                           "NFC canonical reordering and composition");
        m3_expect_encoding(test, &tokenizer, blocked_input,
                           sizeof(blocked_input) - 1U, blocked,
                           sizeof(blocked) / sizeof(blocked[0]),
                           "NFC blocked composition");
        m3_expect_encoding(test, &tokenizer, arabic_input,
                           sizeof(arabic_input) - 1U, arabic_digits,
                           sizeof(arabic_digits) / sizeof(arabic_digits[0]),
                           "Unicode number isolation");
        m3_expect_encoding(test, &tokenizer, nbsp_hi,
                           sizeof(nbsp_hi) - 1U, nbsp_letter,
                           sizeof(nbsp_letter) / sizeof(nbsp_letter[0]),
                           "Unicode whitespace backtracking before letter");
        m3_expect_encoding(test, &tokenizer, nbsp_bang,
                           sizeof(nbsp_bang) - 1U, nbsp_symbol,
                           sizeof(nbsp_symbol) / sizeof(nbsp_symbol[0]),
                           "Unicode whitespace boundary before symbol");
    }
    m3_tokenizer_fixture_dispose(&fixture);
    m3_tokenizer_dispose(&tokenizer);
}

void m3_test_tokenizer_whitespace_vectors(m3_test_context *test)
{
    static const uint32_t spaces_letter[] = {
        (uint32_t)' ', M3_FIXTURE_SPACE_H, (uint32_t)'e', (uint32_t)'l',
        (uint32_t)'l', (uint32_t)'o',
    };
    static const uint32_t spaces_symbol[] = {
        (uint32_t)' ', M3_FIXTURE_SPACE_BANG, (uint32_t)'!',
    };
    static const uint32_t spaces_number[] = {
        (uint32_t)' ', (uint32_t)' ', (uint32_t)'1', (uint32_t)'2',
    };
    static const uint32_t mixed_letter[] = {
        (uint32_t)' ', M3_FIXTURE_TAB_H, (uint32_t)'i',
    };
    static const uint32_t mixed_symbol[] = {
        (uint32_t)' ', (uint32_t)'\t', (uint32_t)'!', (uint32_t)'!',
    };
    static const uint32_t mixed_number[] = {
        (uint32_t)' ', (uint32_t)'\t', (uint32_t)'1',
    };
    static const uint32_t newline_letter[] = {
        (uint32_t)' ', M3_FIXTURE_CRLF, M3_FIXTURE_SPACE_H,
        (uint32_t)'i',
    };
    static const uint32_t symbol_newline[] = {
        (uint32_t)'!', M3_FIXTURE_CRLF,
    };
    m3_tokenizer tokenizer;
    m3_tokenizer_fixture fixture;

    if (m3_load_encoding_fixture(test, &tokenizer, &fixture)) {
        m3_expect_encoding(test, &tokenizer, "  hello", 7U, spaces_letter,
                           sizeof(spaces_letter) / sizeof(spaces_letter[0]),
                           "two spaces before letter");
        m3_expect_encoding(test, &tokenizer, "  !!", 4U, spaces_symbol,
                           sizeof(spaces_symbol) / sizeof(spaces_symbol[0]),
                           "two spaces before symbol");
        m3_expect_encoding(test, &tokenizer, "  12", 4U, spaces_number,
                           sizeof(spaces_number) / sizeof(spaces_number[0]),
                           "two spaces before number");
        m3_expect_encoding(test, &tokenizer, " \thi", 4U, mixed_letter,
                           sizeof(mixed_letter) / sizeof(mixed_letter[0]),
                           "mixed whitespace before letter");
        m3_expect_encoding(test, &tokenizer, " \t!!", 4U, mixed_symbol,
                           sizeof(mixed_symbol) / sizeof(mixed_symbol[0]),
                           "mixed whitespace before symbol");
        m3_expect_encoding(test, &tokenizer, " \t1", 3U, mixed_number,
                           sizeof(mixed_number) / sizeof(mixed_number[0]),
                           "mixed whitespace before number");
        m3_expect_encoding(test, &tokenizer, " \r\n hi", 6U,
                           newline_letter,
                           sizeof(newline_letter) /
                               sizeof(newline_letter[0]),
                           "newline alternative before letter");
        m3_expect_encoding(test, &tokenizer, "!\r\n", 3U,
                           symbol_newline,
                           sizeof(symbol_newline) /
                               sizeof(symbol_newline[0]),
                           "symbol alternative absorbs newline");
    }
    m3_tokenizer_fixture_dispose(&fixture);
    m3_tokenizer_dispose(&tokenizer);
}
