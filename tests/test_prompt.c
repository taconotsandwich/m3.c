/* SPDX-License-Identifier: GPL-2.0-only */

#include "test_cases.h"

#include "m3_prompt.h"

#include <stdint.h>
#include <string.h>

#define M3_TEXT(literal) {literal, sizeof(literal) - 1U}

static void m3_expect_prompt(m3_test_context *test, m3_text_view caption,
                             m3_text_view lyrics, const char *expected)
{
    m3_prompt_text prompt;
    m3_tokenizer_bytes bytes;
    m3_error error;

    m3_prompt_text_init(&prompt);
    M3_TEST_EXPECT(test,
                   m3_prompt_build(&prompt, caption, lyrics, &error) ==
                       M3_STATUS_OK,
                   "prompt build status");
    M3_TEST_EXPECT(test, prompt.data != NULL, "prompt allocation");
    M3_TEST_EXPECT(test, prompt.length == strlen(expected),
                   "prompt byte length");
    M3_TEST_EXPECT(test, strcmp(prompt.data, expected) == 0,
                   "prompt byte content");
    M3_TEST_EXPECT(test, prompt.data[prompt.length] == '\0',
                   "prompt convenience terminator");
    bytes = m3_prompt_tokenizer_bytes(&prompt);
    M3_TEST_EXPECT(test,
                   bytes.data == (const uint8_t *)prompt.data &&
                       bytes.length == prompt.length,
                   "tokenizer byte view");
    m3_prompt_text_dispose(&prompt);
    M3_TEST_EXPECT(test, prompt.data == NULL && prompt.length == 0U,
                   "prompt disposal");
}

void m3_test_prompt_exact_contract(m3_test_context *test)
{
    static const char expected[] =
        "<|im_start|><|caption_start|>dream pop"
        "<|caption_end|><|lyrics_start|>[start]\n[verse]\nHello"
        "<|lyrics_end|><|im_end|><|audio_start|>";

    m3_expect_prompt(test, (m3_text_view)M3_TEXT("dream pop"),
                     (m3_text_view)M3_TEXT("[Verse]\nHello"), expected);
}

void m3_test_prompt_caption_normalization(m3_test_context *test)
{
    static const char expected[] =
        "<|im_start|><|caption_start|>genre is dream pop\nWarm vocals\n"
        "Spacious\nend\nsolo\n<|broken"
        "<|caption_end|><|lyrics_start|>[start]\nwords"
        "<|lyrics_end|><|im_end|><|audio_start|>";
    static const char caption[] =
        "## <|  genre   dream pop  |>\r\n"
        "- **Warm** *vocals*    \r\n"
        "---\r\n"
        "\xE2\x80\xA2 Spacious\r\n\r\n"
        "    end\r\n"
        "<|solo|>\r\n"
        "<|broken";

    m3_expect_prompt(test, (m3_text_view){caption, sizeof(caption) - 1U},
                     (m3_text_view)M3_TEXT("words"), expected);
}

void m3_test_prompt_lyrics_normalization(m3_test_context *test)
{
    static const char expected[] =
        "<|im_start|><|caption_start|>song"
        "<|caption_end|><|lyrics_start|>[start]\n"
        "[arbitrary tag]\n[duet][male]\n"
        "plain\n[bridge]\nleft\nright\n"
        "[unclosed tag\n[]\nmalformed"
        "<|lyrics_end|><|im_end|><|audio_start|>";
    static const char lyrics[] =
        "\t[ARBITRARY TAG] trailing lyric is dropped\r\n"
        "[DUET][Male] also dropped\r"
        "plain [BRIDGE]\r\n"
        "left ^ right\r\n"
        "[Unclosed TAG\r\n"
        "[] malformed";

    m3_expect_prompt(test, (m3_text_view)M3_TEXT("song"),
                     (m3_text_view){lyrics, sizeof(lyrics) - 1U}, expected);
}

void m3_test_prompt_edge_behavior(m3_test_context *test)
{
    static const char expected[] =
        "<|im_start|><|caption_start|>Heading\n<|a|b|>\n"
        "  leading\ntrimmed\n**a*b**"
        "<|caption_end|><|lyrics_start|>[start]\n"
        "[verse]\n \t[hook]\n[Unclosed\n\n trailing \n"
        "<|lyrics_end|><|im_end|><|audio_start|>";
    static const char caption[] =
        "\t## Heading\r\n<|a|b|>\r\n  leading\r\n*trimmed *\r\n"
        "**a*b**\r\n";
    static const char lyrics[] =
        "[VERSE]  \t[HOOK] discarded\r\n[Unclosed\r\n\r\n trailing \r\n";

    m3_expect_prompt(test, (m3_text_view){caption, sizeof(caption) - 1U},
                     (m3_text_view){lyrics, sizeof(lyrics) - 1U}, expected);
}

void m3_test_prompt_trailing_line_breaks(m3_test_context *test)
{
    static const char expected[] =
        "<|im_start|><|caption_start|>caption"
        "<|caption_end|><|lyrics_start|>[start]\nlyrics\n"
        "<|lyrics_end|><|im_end|><|audio_start|>";

    m3_expect_prompt(test, (m3_text_view)M3_TEXT("caption\r\n"),
                     (m3_text_view)M3_TEXT("lyrics\r\n"), expected);
}

void m3_test_prompt_invalid_inputs(m3_test_context *test)
{
    static const char embedded_nul[] = {'a', '\0', 'b'};
    m3_prompt_text prompt;
    m3_error error;
    m3_status status;

    m3_prompt_text_init(&prompt);
    status = m3_prompt_build(&prompt, (m3_text_view)M3_TEXT(" \r\n\t "),
                             (m3_text_view)M3_TEXT("lyrics"), &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_ARGUMENT,
                   "reject empty caption");
    M3_TEST_EXPECT(test, prompt.data == NULL && error.message[0] != '\0',
                   "empty caption structured error");

    status = m3_prompt_build(&prompt, (m3_text_view)M3_TEXT("caption"),
                             (m3_text_view)M3_TEXT("\r\n "), &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_ARGUMENT,
                   "reject empty lyrics");

    status = m3_prompt_build(
        &prompt, (m3_text_view){embedded_nul, sizeof(embedded_nul)},
        (m3_text_view)M3_TEXT("lyrics"), &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_FORMAT,
                   "reject embedded NUL");

    status = m3_prompt_build(&prompt, (m3_text_view){NULL, 4U},
                             (m3_text_view)M3_TEXT("lyrics"), &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_ARGUMENT,
                   "reject null input data");

    status = m3_prompt_build(NULL, (m3_text_view)M3_TEXT("caption"),
                             (m3_text_view)M3_TEXT("lyrics"), &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_INVALID_ARGUMENT,
                   "reject null output");

    status = m3_prompt_build(
        &prompt, (m3_text_view){"x", SIZE_MAX},
        (m3_text_view)M3_TEXT("lyrics"), &error);
    M3_TEST_EXPECT(test, status == M3_STATUS_OVERFLOW,
                   "reject impossible input length before scanning");
    M3_TEST_EXPECT(test,
                   m3_prompt_tokenizer_bytes(NULL).data == NULL &&
                       m3_prompt_tokenizer_bytes(NULL).length == 0U,
                   "null tokenizer byte view");
}
