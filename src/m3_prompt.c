/* SPDX-License-Identifier: GPL-2.0-only */

#include "m3_prompt_internal.h"

#include <stdlib.h>
#include <string.h>

static const char m3_prompt_prefix[] =
    "<|im_start|><|caption_start|>";
static const char m3_caption_suffix[] =
    "<|caption_end|><|lyrics_start|>";
static const char m3_prompt_suffix[] =
    "<|lyrics_end|><|im_end|><|audio_start|>";

void m3_prompt_text_init(m3_prompt_text *prompt)
{
    if (prompt != NULL) {
        prompt->data = NULL;
        prompt->length = 0U;
    }
}

void m3_prompt_text_dispose(m3_prompt_text *prompt)
{
    if (prompt != NULL) {
        free(prompt->data);
        m3_prompt_text_init(prompt);
    }
}

m3_tokenizer_bytes m3_prompt_tokenizer_bytes(const m3_prompt_text *prompt)
{
    m3_tokenizer_bytes bytes = {0};

    if (prompt != NULL) {
        bytes.data = (const uint8_t *)prompt->data;
        bytes.length = prompt->length;
    }
    return bytes;
}

m3_status m3_prompt_build(m3_prompt_text *prompt, m3_text_view caption,
                          m3_text_view lyrics, m3_error *error)
{
    m3_prompt_text normalized_caption;
    m3_prompt_text clean_caption;
    m3_prompt_text normalized_lyrics;
    m3_prompt_text clean_lyrics;
    m3_text_builder builder = {0};
    m3_status status;

    if (prompt == NULL) {
        return m3_error_set(error, M3_STATUS_INVALID_ARGUMENT,
                            "prompt output is null");
    }
    m3_prompt_text_init(prompt);
    m3_prompt_text_init(&normalized_caption);
    m3_prompt_text_init(&clean_caption);
    m3_prompt_text_init(&normalized_lyrics);
    m3_prompt_text_init(&clean_lyrics);

    status = m3_normalize_input(caption, "caption", &normalized_caption,
                                error);
    if (status == M3_STATUS_OK) {
        status = m3_prompt_clean_caption(&normalized_caption, &clean_caption,
                                         error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_normalize_input(lyrics, "lyrics", &normalized_lyrics,
                                    error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_prompt_clean_lyrics(&normalized_lyrics, &clean_lyrics,
                                        error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_append(&builder, m3_prompt_prefix,
                                   sizeof(m3_prompt_prefix) - 1U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_append(&builder, clean_caption.data,
                                   clean_caption.length, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_append(&builder, m3_caption_suffix,
                                   sizeof(m3_caption_suffix) - 1U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_append(&builder, clean_lyrics.data,
                                   clean_lyrics.length, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_append(&builder, m3_prompt_suffix,
                                   sizeof(m3_prompt_suffix) - 1U, error);
    }
    if (status == M3_STATUS_OK) {
        status = m3_builder_finish(&builder, prompt, error);
    }

    m3_prompt_text_dispose(&normalized_caption);
    m3_prompt_text_dispose(&clean_caption);
    m3_prompt_text_dispose(&normalized_lyrics);
    m3_prompt_text_dispose(&clean_lyrics);
    if (status != M3_STATUS_OK) {
        m3_builder_dispose(&builder);
        m3_prompt_text_dispose(prompt);
        return status;
    }
    m3_error_reset(error);
    return M3_STATUS_OK;
}
